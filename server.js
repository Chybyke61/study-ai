require("dotenv").config();

const express = require("express");
const cors = require("cors");
const EventEmitter = require("events");
const progressEvents = new EventEmitter();
const multer = require("multer");
const fs = require("fs");
const path = require("path");
const pdfParse = require("pdf-parse");
const Groq = require("groq-sdk");
const natural = require("natural");
const { createClient } = require('@supabase/supabase-js');
const { S3Client, PutObjectCommand, GetObjectCommand } = require("@aws-sdk/client-s3");
const { getSignedUrl } = require("@aws-sdk/s3-request-presigner");
const { pipeline, max } = require("@xenova/transformers");
const e = require("express");
const mammoth = require("mammoth");
const { type } = require("os");

// --- INITIALIZATION ---
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY);

const r2 = new S3Client({
    region: "auto",
    endpoint: `https://${process.env.R2_ACCOUNT_ID}.r2.cloudflarestorage.com`,
    credentials: {
        accessKeyId: process.env.R2_ACCESS_KEY_ID,
        secretAccessKey: process.env.R2_SECRET_ACCESS_KEY
    }
});

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
const app = express();

app.use(cors({ origin: "*", methods: ["GET", "POST", "DELETE"], allowedHeaders: ["Content-Type", "x-user-id"] }));
app.use(express.json({ limit: "500mb" }));
app.use(express.urlencoded({ extended: true, limit: "500mb" }));

const UPLOAD_DIR = path.join(__dirname, "uploads");
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR);

const CACHE_FILE = path.join(__dirname, "rag_cache.json");

// --- IN-MEMORY STATE ---
let documentStore = {};
let keywordIndices = {};
let vectorIndices = {};
let embedder;

// Load state from disk
if (fs.existsSync(CACHE_FILE)) {
    try {
        const data = JSON.parse(fs.readFileSync(CACHE_FILE, "utf-8"));
        documentStore = data.documentStore || {};
        keywordIndices = data.keywordIndices || {};
        vectorIndices = data.vectorIndices || {};
        console.log("✅ Cache loaded.");
    } catch (err) {
        console.error("❌ Cache read error:", err);
    }
}

function saveCache() {
    setImmediate(() => {
        try {
            fs.writeFileSync(CACHE_FILE, JSON.stringify({ documentStore, keywordIndices, vectorIndices }));
            console.log("💾 Cache auto-saved.");
        } catch (err) {
            console.error("Cache save failed", err);
        }
    });
}

// --- LOAD LOCAL EMBEDDING MODEL ---
(async function initLocalModel() {
    console.log("Loading local embedding model...");
    try {
        // Quantized ensures minimal RAM usage for broad device compatibility
        embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", {
            quantized: true 
        });
        console.log("✅ Local embedding model ready.");
    } catch (err) {
        console.error("❌ Failed to load local model:", err);
    }
})();

// --- HELPER FUNCTIONS ---

function isGoodText(text) {
    if (!text) return false;

    const clean = text.trim();

    return (
        clean.length > 100 &&
        /[a-zA-Z]/.test(clean) && // has letters
        clean.split(" ").length > 20 // enough words
    );
}

function isLikelyScanned(text) {
    if (!text) return true;

    const clean = text.trim();

    // Too short or no real words → likely scanned
    return (
        clean.length < 50 ||
        !/[a-zA-Z]{3,}/.test(clean)
    );
}

async function extractText(file) {
    const ext = path.extname(file.path).toLowerCase();
    try {
        // ✅ PDF
        if (ext === ".pdf") {
        console.log("📄 Parsing PDF...");

        const buffer = fs.readFileSync(file.path);
        const data = await pdfParse(buffer);

        const text = data.text || "";

        console.log("PDF text length:", text.length);
        console.log("Preview:", text.slice(0, 100));

        // 🔥 ACCEPT ANY REAL TEXT
        if (text.trim().length > 20) {
            console.log("✅ PDF parsed successfully");
            return text;
        }

        console.warn("⚠️ PDF has very little text");
        return "";

    } 
        
 // ======================
// ✅ DOCX FIX (PASTE HERE)
// ======================
        if (ext === ".docx") {
           console.log("📄 Reading DOCX with mammoth...");

           const result = await mammoth.extractRawText({
                path: file.path
          });

           console.log("DOCX text length:", result.value.length);

           if (result.value && result.value.trim().length > 50) {
           console.log("✅ DOCX parsed");
           return result.value;
        }

        console.warn("⚠️ DOCX empty");
        return "";
        }
    } catch (err) {
        console.error("❌ Extraction failed:", err);
        return "";
    }
}

function recursiveChunk(text, chunkSize = 1000, overlap = 200) {
    const words = text.split(/\s+/);
    const chunks = [];
    for (let i = 0; i < words.length; i += (chunkSize - overlap)) {
        chunks.push(words.slice(i, i + chunkSize).join(" "));
    }
    return chunks;
}

async function embedText(text) {
    if (!embedder) {
        console.warn("Embedder is still loading, returning zero-vector.");
        return new Array(384).fill(0);
    }

    try {
        const result = await embedder(text, { pooling: "mean", normalize: true });
        return Array.from(result.data);
    } catch (err) {
        console.error("Local Embedding Error:", err);
        return new Array(384).fill(0);
    }
}

async function addToIndex(userId, filename, children) {
    if (!keywordIndices[userId]) keywordIndices[userId] = {};
    if (!vectorIndices[userId]) vectorIndices[userId] = {};

    const MAX_CHUNKS = 100;
children = children.slice(0, MAX_CHUNKS);

    const tfidf = new natural.TfIdf();
    const vectors = [];
    
    // Strict batch size prevents buffer overflow on constrained hardware
    const batchSize = 50; 

    children.forEach(chunk => {
    if (!chunk || typeof chunk !== "string") return;
    tfidf.addDocument(chunk.toLowerCase());
});

    for (let i = 0; i < children.length; i += batchSize) {
        try {
            const batch = children
                .slice(i, i + batchSize)
                .filter(t => typeof t === "string" && t.trim().length > 20);

            const batchVectors = await Promise.all(
                batch
                    .filter(t => typeof t === "string" && t.trim().length > 20)
                    .map(async text => ({
                        text,
                        vector: await embedText(text.toLowerCase())
                    }))
            );

            vectors.push(...batchVectors);
            console.log(`[${filename}] Indexed ${vectors.length}/${children.length} chunks locally...`);

        } catch (err) {
            console.error(`Batch starting at ${i} failed for ${filename}`, err);
        }
    }

    keywordIndices[userId][filename] = tfidf;
    vectorIndices[userId][filename] = vectors;
    saveCache();
}

function cosineSimilarity(vecA, vecB) {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }
    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

/* ADD STEP 3 HERE */

async function rebuildIndexesFromSupabase() {

    console.log("Rebuilding AI indexes from Supabase...");

    const { data, error } = await supabase
        .from("book_chunks")
        .select("*");

    if (error) {
        console.error("Failed to load chunks:", error);
        return;
    }

    for (const row of data) {

        const userId = row.user_id;
        const book = row.filename;
        const text = row.content;

        if (!userId || !book || !text) continue;

        if (!documentStore[userId]) documentStore[userId] = {};
        if (!documentStore[userId][book]) {
            documentStore[userId][book] = { childChunks: [] };
        }

        documentStore[userId][book].childChunks.push(text);
    }

    console.log("Chunks loaded from Supabase");

    for (const userId in documentStore) {
        for (const book in documentStore[userId]) {

            const chunks = documentStore[userId][book].childChunks;

            await addToIndex(userId, book, chunks);

            console.log("Index rebuilt for", book);
        }
    }

    console.log("AI indexes rebuilt successfully");
}

async function rerankChunks(query, chunks) {
    try {
        const limitedChunks = chunks.slice(0, 5); // 🔥 limit for free tier

        const prompt = `
Select the 5 most relevant chunks for the query.

Query:
${query}

Chunks:
${limitedChunks.map((c, i) => `[${i}] ${c}`).join("\n\n")}

Return ONLY numbers like: 2,5,1,3,0
`;

        const response = await groq.chat.completions.create({
            model: "llama-3.1-8b-instant",
            messages: [{ role: "user", content: prompt }],
            temperature: 0
        });

        const text = response.choices[0].message.content;

        const indices = text.match(/\d+/g)?.map(Number) || [];

        return indices
            .map(i => limitedChunks[i])
            .filter(Boolean);

    } catch (err) {
        console.error("Rerank failed:", err);
        return chunks.slice(0, 5); // fallback
    }
}

// --- ROUTES ---

app.post("/generate-upload-url", async (req, res) => {
    try {
        const userId = req.headers["x-user-id"];
        const { filename } = req.body;

        if (!userId || !filename) {
            return res.status(400).json({ error: "Missing filename or user." });
        }

        const key = `${userId}/${Date.now()}_${filename}`;

        const command = new PutObjectCommand({
            Bucket: process.env.R2_BUCKET,
            Key: key,
            ContentType: "application/octet-stream"
        });

        const uploadUrl = await getSignedUrl(r2, command, { expiresIn: 600 });

        res.json({
            uploadUrl,
            fileKey: key
        });

    } catch (err) {
        console.error("Signed URL generation failed:", err);
        res.status(500).json({ error: "Could not generate upload URL." });
    }
});

app.post("/upload", async (req, res) => {
    let tempPath = null;
    try {
        console.log("UPLOAD ROUTE HIT");
        const userId = req.headers["x-user-id"];
        const { fileKey, filename } = req.body;

        console.log("User:", userId);
        console.log("File Key:", fileKey);
        console.log("Filename:", filename);

        if (!userId || !fileKey || !filename) {
            return res.status(400).json({ error: "Missing upload data." });
        }

        tempPath = path.join(UPLOAD_DIR, Date.now() + "-" + filename);

        // 1. Stream from R2
        const command = new GetObjectCommand({
            Bucket: process.env.R2_BUCKET,
            Key: fileKey
        });
        const response = await r2.send(command);

        await new Promise((resolve, reject) => {
           const writeStream = fs.createWriteStream(tempPath);

           response.Body.on("error", reject);
           writeStream.on("error", reject);

           writeStream.on("finish", () => {
           console.log("✅ File fully downloaded");
           resolve();

        progressEvents.emit("update", { step: "Uploading...", progress: 10 });
    });

          response.Body.pipe(writeStream);
 });
        const stats = fs.statSync(tempPath);
         console.log("📦 File size:", stats.size);
        
        // 2. Extract Text
        console.log("STEP 1: Starting upload");
        const text = await extractText({ path: tempPath });
        progressEvents.emit("update", { step: "Extracting...", progress: 30 });
        console.log("STEP 2: Extracted text length:", text ? text.length : 0);

        // CRITICAL: Stop if extraction failed
        if (!text || text.trim().length < 50) {
            if (fs.existsSync(tempPath)) fs.unlinkSync(tempPath);
            return res.status(422).json({ error: "Could not extract study material from this file." });
        }

        // 3. Chunking
        const parentChunks = recursiveChunk(text, 1500, 200);
        const childChunks = recursiveChunk(text, 700, 100);
        progressEvents.emit("update", { step: "Chunking...", progress: 50 });

        // 🚀 LIMIT chunks (IMPORTANT)
        const MAX_CHUNKS = 100;
        const limitedChunks = childChunks.slice(0, MAX_CHUNKS);
        console.log("STEP 3: Chunks created:", childChunks.length);

        if (!documentStore[userId]) documentStore[userId] = {};
        documentStore[userId][filename] = { parentChunks, childChunks };
     

        console.log("STEP 4: Starting embedding...");

        // Save chunks to Supabase so they survive redeploy
console.log(`⚡️ Embedding ${limitedChunks.length} chunks...`);

const batchSize = 50;
const insertBatch = [];

for (let i = 0; i < limitedChunks.length; i += batchSize) {

    const batch = limitedChunks.slice(i, i + batchSize);

    const vectors = await Promise.all(
        batch.map(chunk => embedText(chunk))
    );

    vectors.forEach((vector, idx) => {
        insertBatch.push({
            user_id: userId,
            filename,
            content: batch[idx],
            embedding: vector
        });
    });

    const progress = 60 + Math.floor(((i + batch.length) / limitedChunks.length) * 30);

    progressEvents.emit("update", {
    step: "Embedding...",
    progress
});

    console.log(`📊 Progress: ${Math.min(i + batch.length, limitedChunks.length)}/${limitedChunks.length}`);
}

// ONE INSERT (FAST 🚀)
const { error } = await supabase
    .from("book_chunks")
    .insert(insertBatch);

progressEvents.emit("update", { step: "Saving...", progress: 90 });

if (error) {
    console.error("❌ Supabase batch insert error:", error);
    throw new Error("Failed saving chunks");
}

console.log("✅ All chunks saved successfully");

        // 4. Update Database
        const { error: dbError } = await supabase
            .from("books")
            .insert([{ user_id: userId, filename }]);

        if (dbError) throw dbError;

        // 5. Indexing in background (Non-blocking)
        setImmediate(async () => {
            try {
                await addToIndex(userId, filename, childChunks);
                console.log(`✅ Indexing complete for: ${filename}`);
            } catch (err) {
                console.error("Indexing failed in background:", err);
            } finally {
                // Safely delete temp file ONLY after extraction is confirmed done
                if (fs.existsSync(tempPath)) fs.unlinkSync(tempPath);
            }
        });

        progressEvents.emit("update", { step: "Completed ✅", progress: 100 });

        // 6. Respond immediately to Frontend
        res.json({ success: true, name: filename });

    } catch (err) {
        console.error("Upload processing error:", err);

        if (fs.existsSync(tempPath)) {
            try { fs.unlinkSync(tempPath); } catch {}

        }
        res.status(500).json({ error: "Upload failed. Please try again." });

}
});

function buildSmartContext(chunks, maxChars = 2500) {
  let context = "";
  let used = 0;

  for (const chunk of chunks) {
    if (!chunk || typeof chunk !== "string") continue;

    const clean = chunk.trim();

    if (used + clean.length > maxChars) break;

    context += clean + "\n\n---\n\n";
    used += clean.length;
  }

  return context.trim();
}

function reciprocalRankFusion(vectorResults, keywordResults, k = 60) {
  const scores = new Map();

  function add(results) {
    results.forEach((item, index) => {
      const key = item.text;

      const rankScore = 1 / (k + index);

      if (!scores.has(key)) {
        scores.set(key, { text: item.text, score: 0 });
      }

      scores.get(key).score += rankScore;
    });
  }

  add(vectorResults);
  add(keywordResults);

  return Array.from(scores.values())
    .sort((a, b) => b.score - a.score)
    .map(item => item.text);
}

app.post("/deep-explain", async (req, res) => {
    try {
        const { topic, level = "University", book = "all" } = req.body;
        const userId = req.headers["x-user-id"];

        // 🔍 CHECK CACHE FIRST
        const { data: cached } = await supabase
            .from("ai_cache")
            .select("response")
            .eq("user_id", userId)
            .eq("topic", topic)
            .eq("level", level)
            .eq("book", book)
            .eq("type", "explain")
            .limit(1)
            .single();

        if (cached) {
            console.log("⚡️ Cache hit");
            return res.json({ explanation: cached.response });
        }

        if (!topic) {
            return res.status(400).json({ error: "Topic required" });
        }

        const queryVector = await embedText(topic.toLowerCase());

            const { data, error } = await 
            supabase.rpc("match_book_chunks", {
            query_embedding: queryVector,
            match_threshold: 0,
            match_count: 5,
            p_user_id: userId,
            p_filename: book === "all" ? null : book
        });

        let keywordResults = [];

if (keywordIndices[userId] && keywordIndices[userId][book]) {

  const tfidf = keywordIndices[userId][book];

  tfidf.tfidfs(topic, function(i, measure) {
    keywordResults.push({
      score: measure,
      text: documentStore[userId][book].childChunks[i]
    });
  });

  keywordResults.sort((a,b)=>b.score-a.score);

  keywordResults = keywordResults.slice(0,3);

}

const vectorContext = data ? data.map(row => row.content) : [];
      
const vectorFormatted = vectorContext.map(text => ({ text }));

const keywordFormatted = keywordResults.map(r => ({
  text: r.text
}));
let combinedContext;

if (keywordFormatted.length > 0) {
  combinedContext = reciprocalRankFusion(
    vectorFormatted,
    keywordFormatted
  );
} else {
  combinedContext = vectorContext;
}


        console.log("Vector search results:", data);
        console.log("Vector results:", vectorContext.length);
        console.log("Hybrid retrieval working:");
console.log("Keyword results:", keywordResults.length);
console.log("Combined results:", combinedContext.length);

        if (error) {
    console.error("Vector search error:", error);
}
        if (!data || data.length === 0) {
  return res.json({ explanation: "No study material found for this topic." });
}

// Extract raw chunks
const rawChunks = combinedContext;

// 🔥 Smart rerank (only when needed)
let bestChunks;

if (rawChunks.length > 8) {
    bestChunks = await rerankChunks(topic, rawChunks);
} else {
    bestChunks = rawChunks;
}

// Build final context
// 🔥 LIMIT CONTEXT SIZE (CRITICAL FIX)
const context = buildSmartContext(bestChunks, 2200);

        const prompt = `
You are an expert academic tutor for all subjects (science, medicine, engineering, humanities, business, etc.).

STRICT RULES:
- You MUST use ONLY the provided textbook context
- You are NOT allowed to use outside knowledge
- If the answer is not clearly contained in the context, respond exactly with:
  "This is not found in the provided material"
- Do NOT guess, infer beyond the text, or hallucinate

INSTRUCTIONS:
- Do NOT greet the user
- Do NOT say "Welcome", "Hello", or address the student directly
- Start immediately with the explanation
- Do NOT include a "Definition" section
- Organize the explanation using clear headings
- Each heading must explain a key concept
- Provide deep, detailed explanations under each heading
- Explain mechanisms, processes, and relationships clearly
- Only include examples if they are supported by the context
- Ensure the explanation is understandable without needing the original textbook
- Use clear academic language suitable for university-level learning
- Avoid repetition and vague statements

---------------------
TEXTBOOK CONTEXT:
${context}
---------------------

TOPIC:
${topic}

TASK:
Provide a structured, detailed explanation using headings and well-developed paragraphs strictly based on the context.
`;
    
        const chat = await groq.chat.completions.create({
            messages: [{ role:"user", content:prompt }],
            model: "llama-3.1-8b-instant",
            temperature: 0.2,
            max_tokens: 1500
        });

        const output = chat.choices[0].message.content;

        // 💾 SAVE TO CACHE
        await supabase.from("ai_cache").insert({
            user_id: userId,
            topic,
            level,
            book,
            type: "explain",
            response: output 

        });
     
        res.json({ explanation: output });

    } catch(err) {
        console.error("Explain error:", err);
        res.status(500).json({ error:"Explain failed" });
    }
});
app.post("/notes", async (req, res) => {
    try {

        const { topic, level, book } = req.body;
        const userId = req.headers["x-user-id"];
        // 🔍 CHECK CACHE FIRST
        const { data: cached } = await supabase
            .from("ai_cache")
            .select("response")
            .eq("user_id", userId)
            .eq("topic", topic)
            .eq("level", level)
            .eq("book", book)
            .eq("type", "notes")
            .limit(1)
            .single();

        if (cached) {
            console.log("⚡️ Notes cache hit");
            return res.json({ notes: cached.response });
        }

        const queryVector = await embedText(topic.toLowerCase());

        let results = [];
        let booksToSearch = [];

        if (book === "all") {
            booksToSearch = Object.keys(vectorIndices[userId] || {});
        } else {
            booksToSearch = [book];
        }

        for (const b of booksToSearch) {

            const vectors = vectorIndices[userId]?.[b];
            const chunks = documentStore[userId]?.[b]?.childChunks;

            if (!vectors || !chunks) continue;

            vectors.forEach((vecObj,i)=>{
                const score = cosineSimilarity(queryVector, vecObj.vector);
                results.push({score,text:chunks[i]});
            });
        }

        if (results.length === 0) {
            return res.json({ notes:"No study material found." });
        }

        results.sort((a,b)=>b.score-a.score);

        let keywordResults = [];

if (keywordIndices[userId] && keywordIndices[userId][book]) {
  const tfidf = keywordIndices[userId][book];

  tfidf.tfidfs(topic, function(i, measure) {
  const chunk = documentStore[userId]?.[book]?.childChunks?.[i];

  if (chunk && typeof chunk === "string") {
    keywordResults.push({
      score: measure,
      text: chunk
    });
  }
});

  keywordResults.sort((a,b)=>b.score-a.score);
  keywordResults = keywordResults.slice(0, 3);
}

        // Extract raw chunks
        const vectorContext = results.map(r => r.text);

const vectorFormatted = vectorContext.map(text => ({ text }));

const keywordFormatted = keywordResults.map(r => ({
  text: r.text
}));

let combinedContext;

if (keywordFormatted.length > 0) {
  combinedContext = reciprocalRankFusion(
    vectorFormatted,
    keywordFormatted
  );
} else {
  combinedContext = vectorContext;
}

const rawChunks = combinedContext;

        // 🔥 Smart rerank
        let bestChunks;

        if (rawChunks.length > 8) {
            bestChunks = await rerankChunks(topic, rawChunks);
    } else {
            bestChunks = rawChunks;
    }

    // Build context
    const context = buildSmartContext(bestChunks, 2200);

        const prompt = `
You are an expert academic tutor creating high-quality study notes for any subject.

STRICT RULES:
- You MUST use ONLY the provided textbook context
- You are NOT allowed to use outside knowledge
- If the topic is not clearly covered in the context, respond exactly with:
  "This is not found in the provided material"
- Do NOT guess or hallucinate

INSTRUCTIONS:
- Do NOT greet the user
- Do NOT say "Welcome" or "Hello"
- Write clear, well-structured study notes
- Use concise but detailed explanations
- Avoid repetition and filler text
- Ensure notes are easy to revise and memorize
- Use bullet points where appropriate

---------------------
TEXTBOOK CONTEXT:
${context}
---------------------

TOPIC:
${topic}

STRUCTURE:

# Topic Overview
- Provide a clear explanation of the topic based strictly on the context

# Key Concepts
- List and explain the most important ideas from the context

# Mechanisms / Processes
- Explain step-by-step processes or how things work (if applicable)

# Key Points Summary
- Provide concise bullet points for quick revision

# Exam Tips
- Highlight likely exam areas based ONLY on the context

TASK:
Create detailed, structured study notes strictly based on the provided context.
`;


        const chat = await groq.chat.completions.create({
            messages:[{role:"user",content:prompt}],
            model:"llama-3.1-8b-instant",
            temperature: 0.2,
            max_tokens: 1500
        });

        const output = chat.choices[0].message.content;

        // 💾 SAVE TO CACHE
        await supabase.from("ai_cache").insert({
            user_id: userId,
            topic,
            level,
            book,
            type: "notes",
            response: output
        });

        res.json({ notes: output });

    } catch(err){
        console.error("Notes error:", err);
        res.status(500).json({ error:"Notes failed" });
    }
});
  app.post("/quiz", async (req, res) => {
    try {

        const { topic, level, book } = req.body;
        const userId = req.headers["x-user-id"];

        // 🔍 CHECK CACHE FIRST
        const { data: cached } = await supabase
            .from("ai_cache")
            .select("response")
            .eq("user_id", userId)
            .eq("topic", topic)
            .eq("level", level)
            .eq("book", book)
            .eq("type", "quiz")
            .limit(1)
            .single();

        if (cached) {
            console.log("⚡️ Quiz cache hit");
            return res.json({ quiz: cached.response });
        }

        const queryVector = await embedText(topic.toLowerCase());

        let results = [];
        let booksToSearch = [];

        if (book === "all") {
            booksToSearch = Object.keys(vectorIndices[userId] || {});
        } else {
            booksToSearch = [book];
        }

        for (const b of booksToSearch) {

            const vectors = vectorIndices[userId]?.[b];
            const chunks = documentStore[userId]?.[b]?.childChunks;

            if (!vectors || !chunks) continue;

            vectors.forEach((vecObj,i)=>{
                const score = cosineSimilarity(queryVector, vecObj.vector);
                results.push({score,text:chunks[i]});
            });
        }

        if (results.length === 0) {
            return res.json({ quiz:"No study material found." });
        }

        results.sort((a,b)=>b.score-a.score);

        let keywordResults = [];

if (keywordIndices[userId] && keywordIndices[userId][book]) {
  const tfidf = keywordIndices[userId][book];

  tfidf.tfidfs(topic, function(i, measure) {
    const chunk = documentStore[userId]?.[book]?.childChunks?.[i];

    if (chunk && typeof chunk === "string") {
      keywordResults.push({
        score: measure,
        text: chunk
      });
    }
  });

  keywordResults.sort((a,b)=>b.score-a.score);
  keywordResults = keywordResults.slice(0, 3);
}

        // Extract raw chunks
        const vectorContext = results.map(r => r.text);

const vectorFormatted = vectorContext.map(text => ({ text }));

const keywordFormatted = keywordResults.map(r => ({
  text: r.text
}));

let combinedContext;

if (keywordFormatted.length > 0) {
  combinedContext = reciprocalRankFusion(
    vectorFormatted,
    keywordFormatted
  );
} else {
  combinedContext = vectorContext;
}

const rawChunks = combinedContext;

        // 🔥 Smart rerank
        let bestChunks;

        if (rawChunks.length > 8) {
        bestChunks = await rerankChunks(topic, rawChunks);
    } else {
        bestChunks = rawChunks;
}

        // Build context
        const context = buildSmartContext(bestChunks, 2200);

        const prompt = `
You are an expert academic tutor creating a challenging university-level quiz for any subject.

STRICT RULES:
- You MUST use ONLY the provided textbook context
- You are NOT allowed to use outside knowledge
- Every question MUST be directly based on the context
- If there is not enough information to create the quiz, respond exactly with:
  "This is not found in the provided material"
- Do NOT guess or hallucinate

INSTRUCTIONS:
- Do NOT greet the user
- Do NOT include any introductory text
- Make the questions challenging (application, reasoning, not just recall)
- Avoid repeating the same concept
- Ensure options are realistic and not obvious
- Keep explanations clear and directly tied to the context

---------------------
TEXTBOOK CONTEXT:
${context}
---------------------

TOPIC:
${topic}

REQUIREMENTS:
- Generate EXACTLY 10 multiple choice questions
- Each question must have 4 options (A–D)
- Only ONE correct answer per question
- Provide the correct answer
- Provide a short explanation based ONLY on the context

FORMAT:

Question 1  
A)  
B)  
C)  
D)  

Correct Answer:  
Explanation:  

(Repeat for all 10 questions)
`;

        const chat = await groq.chat.completions.create({
            messages:[{role:"user",content:prompt}],
            model:"llama-3.1-8b-instant",
            temperature: 0.2,
            max_tokens: 1500
        });
            
        const output = chat.choices[0].message.content;

        // 💾 SAVE TO CACHE
        await supabase.from("ai_cache").insert({
            user_id: userId,
            topic,
            level,
            book,
            type: "quiz",
            response: output
        });

        res.json({ quiz: output });

    } catch(err){
        console.error("Quiz error:", err);
        res.status(500).json({ error:"Quiz failed" });
    }
});

app.post("/chat", async (req, res) => {
    try {
        const { query, history = [], books = [] } = req.body;
        const userId = req.headers["x-user-id"];

        if (!query) return res.status(400).json({ error: "Query is required" });
        if (!books || books.length === 0) return res.status(400).json({ error: "Select at least one book." });

        const queryVector = await embedText(query.toLowerCase());
        let allResults = [];

        for (const book of books) {
            const vectors = vectorIndices[userId]?.[book];
            const childChunks = documentStore[userId]?.[book]?.childChunks;

            if (vectors && childChunks) {
                vectors.forEach((vecObj, index) => {
                    if (index < childChunks.length) {
                        const score = cosineSimilarity(queryVector, vecObj.vector);
                        allResults.push({ score, text: childChunks[index], book });
                    }
                });
            }
        }

        allResults.sort((a, b) => b.score - a.score);

        // Extract raw chunks
        const rawChunks = allResults.map(r => r.text);

        // 🔥 Smart rerank (only if needed)
            let bestChunks;

    if (rawChunks.length > 5) {
    bestChunks = await rerankChunks(query, rawChunks);
    } else {
    bestChunks = rawChunks;
    }

// Build context
        const context = bestChunks.join("\n\n---\n\n");

        const prompt = `You are a helpful study assistant. Use the textbook excerpts to answer the question.\n\nContext:\n${context}\n\nQuestion: ${query}`;

        const chatCompletion = await groq.chat.completions.create({
            messages: [{ role: "user", content: prompt }],
            model: "llama-3.1-8b-instant",
            temperature: 0.5,
        });

        res.json({ answer: chatCompletion.choices[0].message.content });

    } catch (err) {
        console.error("Chat error:", err);
        res.status(500).json({ error: "Chat generation failed." });
    }
});

app.get("/books", async (req, res) => {
    try {
        const userId = req.headers["x-user-id"];
        const { data, error } = await supabase.from("books").select("filename").eq("user_id", userId);
        if (error) return res.status(500).json({ error: "Failed to load books" });
        res.json(data.map(book => ({ name: book.filename })));
    } catch (err) { 
        res.status(500).json({ error: "Server error" }); 
    }
});

app.delete("/delete-book/:name", async (req, res) => {
    const name = decodeURIComponent(req.params.name);
    const userId = req.headers["x-user-id"];

    if (documentStore[userId] && documentStore[userId][name]) {
        delete documentStore[userId][name];
        delete keywordIndices[userId]?.[name];
        delete vectorIndices[userId]?.[name];
        saveCache();
        return res.json({ success: true });
    }
    res.status(404).json({ error: "Not found" });
});

app.get("/health", (req, res) => res.json({ status: "ok" }));

app.get("/progress", (req, res) => {
    res.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive"
    });

    const onProgress = (data) => {
        res.write(`data: ${JSON.stringify(data)}\n\n`);
    };

    progressEvents.on("update", onProgress);

    req.on("close", () => {
        progressEvents.removeListener("update", onProgress);
    });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, async () => {

    console.log(`🚀 Server running on port ${PORT}`);

    try {
        await rebuildIndexesFromSupabase();
    } catch (err) {
        console.error("Startup index rebuild failed:", err);
    }

});
