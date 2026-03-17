require("dotenv").config();

const express = require("express");
const cors = require("cors");
const multer = require("multer");
const fs = require("fs");
const path = require("path");
const textract = require("textract");
const pdfParse = require("pdf-parse");
const Groq = require("groq-sdk");
const natural = require("natural");
const { createClient } = require('@supabase/supabase-js');
const { S3Client, PutObjectCommand, GetObjectCommand } = require("@aws-sdk/client-s3");
const { getSignedUrl } = require("@aws-sdk/s3-request-presigner");
const { pipeline, max } = require("@xenova/transformers");
const e = require("express");

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
        if (ext === ".pdf") {
    const buffer = fs.readFileSync(file.path);
            console.log("📄 Parsing PDF...");
            console.log("Buffer size:", buffer.length);

    // 🔹 STEP 1: Try pdf-parse
    let pdfText = "";
    try {
        const data = await pdfParse(buffer);
        pdfText = data.text;
        console.log("Extracted preview:", pdfText?.slice(0, 100));
    } catch (err) {
        console.warn("pdf-parse failed");
    }

    // 🔥 SMART DECISION
    if (isGoodText(pdfText) && !isLikelyScanned(pdfText)) {
        console.log("✅ Using pdf-parse (clean text)");
        return pdfText;
    }

    console.warn("⚠️ PDF looks scanned, trying textract...");

    // 🔹 STEP 2: Try textract
    let textractText = "";
    try {
        textractText = await new Promise((resolve) => {
    textract.fromFileWithPath(file.path, (err, text) => {
        if (err) {
            console.error("Textract error:", err);
            return resolve("");
        }
        resolve(text);
    });
});
    } catch (err) {
        console.warn("textract failed");
    }

    if (isGoodText(textractText)) {
        console.log("✅ Using textract");
        return textractText;
    }

    console.warn("⚠️ Could not extract readable text from PDF.");
    return "";
}
    } catch (err) {
        console.error("OCR failed:", err);
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

    const tfidf = new natural.TfIdf();
    const vectors = [];
    
    // Strict batch size prevents buffer overflow on constrained hardware
    const batchSize = 20; 

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
        const limitedChunks = chunks.slice(0, 8); // 🔥 limit for free tier

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
    });

          response.Body.pipe(writeStream);
 });
        const stats = fs.statSync(tempPath);
         console.log("📦 File size:", stats.size);
        
        // 2. Extract Text
        console.log("STEP 1: Starting upload");
        const text = await extractText({ path: tempPath });
        console.log("STEP 2: Extracted text length:", text ? text.length : 0);

        // CRITICAL: Stop if extraction failed
        if (!text || text.trim().length < 50) {
            if (fs.existsSync(tempPath)) fs.unlinkSync(tempPath);
            return res.status(422).json({ error: "Could not extract study material from this file." });
        }

        // 3. Chunking
        const parentChunks = recursiveChunk(text, 1500, 200);
        const childChunks = recursiveChunk(text, 400, 50);
        console.log("STEP 3: Chunks created:", childChunks.length);

        if (!documentStore[userId]) documentStore[userId] = {};
        documentStore[userId][filename] = { parentChunks, childChunks };
     

        console.log("STEP 4: Starting embedding...");

        // Save chunks to Supabase so they survive redeploy
console.log(`⚡️ Embedding ${childChunks.length} chunks...`);

const batchSize = 20;
const insertBatch = [];

for (let i = 0; i < childChunks.length; i += batchSize) {

    const batch = childChunks.slice(i, i + batchSize);

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

    console.log(`📊 Progress: ${Math.min(i + batch.length, childChunks.length)}/${childChunks.length}`);
}

// ONE INSERT (FAST 🚀)
const { error } = await supabase
    .from("book_chunks")
    .insert(insertBatch);

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

app.post("/deep-explain", async (req, res) => {
    try {
        const { topic, book } = req.body;
        const userId = req.headers["x-user-id"];

        if (!topic) {
            return res.status(400).json({ error: "Topic required" });
        }

        const queryVector = await embedText(topic.toLowerCase());

            const { data, error } = await 
            supabase.rpc("match_book_chunks", {
            query_embedding: queryVector,
            match_threshold: 0,
            match_count: 8,
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
const keywordContext = keywordResults.map(r => r.text);

const combinedContext = [...vectorContext, ...keywordContext];

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

if (rawChunks.length > 5) {
    bestChunks = await rerankChunks(topic, rawChunks);
} else {
    bestChunks = rawChunks;
}

// Build final context
const context = bestChunks.join("\n\n---\n\n");
    

 const prompt = `
You are an expert academic tutor.

Instructions:
- Do NOT greet the user.
- Do NOT say "Welcome", "Hello", or address the student directly.
- Start immediately with the explanation of the topic.
- Do NOT include a "Definition" section.
- Organize the explanation using clear headings.
- Each heading should explain an important concept related to the topic.
- Write detailed explanations so the student can understand the topic without reading the textbook.
- Expand mechanisms, processes, causes, and relationships thoroughly.
- Use clear academic language suitable for university-level learning.
- Avoid short answers.

Textbook Context:
${context}

Topic:
${topic}

Provide a structured explanation using headings and detailed paragraphs.
`;

        const chat = await groq.chat.completions.create({
            messages: [{ role:"user", content:prompt }],
            model: "llama-3.1-8b-instant",
            temperature: 0.4,
            max_tokens: 1500
        });

        res.json({
            explanation: chat.choices[0].message.content
        });

    } catch(err) {
        console.error("Explain error:", err);
        res.status(500).json({ error:"Explain failed" });
    }
});
app.post("/notes", async (req, res) => {
    try {

        const { topic, book } = req.body;
        const userId = req.headers["x-user-id"];

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

        // Extract raw chunks
        const rawChunks = results.map(r => r.text);

        // 🔥 Smart rerank
        let bestChunks;

        if (rawChunks.length > 5) {
            bestChunks = await rerankChunks(topic, rawChunks);
    } else {
            bestChunks = rawChunks;
    }

    // Build context
    const context = bestChunks.join("\n\n---\n\n");


        const prompt = `
Create **detailed university-level study notes**.

Topic: ${topic}

Textbook Context:
${context}

Structure the notes like this:

# Topic Overview
# Key Definitions
# Important Concepts
# Mechanisms or Processes
# Bullet Point Summary
# Exam Tips

The notes must be detailed and structured for studying.
`;

        const chat = await groq.chat.completions.create({
            messages:[{role:"user",content:prompt}],
            model:"llama-3.1-8b-instant",
        });

        res.json({
            notes: chat.choices[0].message.content
        });

    } catch(err){
        console.error("Notes error:", err);
        res.status(500).json({ error:"Notes failed" });
    }
});
  app.post("/quiz", async (req, res) => {
    try {

        const { topic, book } = req.body;
        const userId = req.headers["x-user-id"];

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

        // Extract raw chunks
        const rawChunks = results.map(r => r.text);

        // 🔥 Smart rerank
        let bestChunks;

        if (rawChunks.length > 5) {
        bestChunks = await rerankChunks(topic, rawChunks);
    } else {
        bestChunks = rawChunks;
}

        // Build context
        const context = bestChunks.join("\n\n---\n\n");

        const prompt = `
Create a **difficult university-level quiz**.

Topic: ${topic}

Textbook Context:
${context}

Requirements:

• 5 multiple choice questions  
• Each question must have 4 options  
• Show the correct answer  
• Provide a short explanation  

Format:

Question 1  
A)  
B)  
C)  
D)  

Correct Answer:  
Explanation:
`;

        const chat = await groq.chat.completions.create({
            messages:[{role:"user",content:prompt}],
            model:"llama-3.1-8b-instant",
        });

        res.json({
            quiz: chat.choices[0].message.content
        });

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

const PORT = process.env.PORT || 3000;
app.listen(PORT, async () => {

    console.log(`🚀 Server running on port ${PORT}`);

    try {
        await rebuildIndexesFromSupabase();
    } catch (err) {
        console.error("Startup index rebuild failed:", err);
    }

});
