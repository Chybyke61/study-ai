require("dotenv").config();

const express = require("express");
const cors = require("cors");
const EventEmitter = require("events");
const progressEvents = new EventEmitter();
const multer = require("multer");
const upload = multer({
    dest: "uploads/",
    limits: { fileSize: 50 * 1024 * 1024 } // 100MB safe
});
const fs = require("fs");
const path = require("path");
const pdfParse = require("pdf-parse");
const Groq = require("groq-sdk");
const natural = require("natural");
const { createClient } = require('@supabase/supabase-js');
const { S3Client, PutObjectCommand, GetObjectCommand, DeleteObjectCommand, ListObjectsV2Command } = require("@aws-sdk/client-s3");
const { getSignedUrl } = require("@aws-sdk/s3-request-presigner");
const { pipeline, max } = require("@xenova/transformers");
const e = require("express");
const mammoth = require("mammoth");
const officeParser = require("officeparser");
const pptx2json = require("pptx2json");
const pptxParser = pptx2json.default || pptx2json;
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

//app.use(cors({ origin: "*", methods: ["GET", "POST", "DELETE"], allowedHeaders: ["Content-Type", "x-user-id"] }));
app.use(cors({
    origin: ["https://studyai-app.vercel.app"],
    methods: ["GET", "POST", "DELETE"],
    allowedHeaders: ["Content-Type", "x-user-id"]
}));
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
    const fileName = file.originalname || file.filename || file.path || "";
    const ext = path.extname(fileName).toLowerCase();

    console.log("📁 File:", fileName);
    console.log("📦 Extension:", ext);

    try {
        // ✅ PDF
        if (ext === ".pdf") {
            const buffer = fs.readFileSync(file.path);
            const data = await pdfParse(buffer);
            return data.text || "";
        }

        // ✅ DOCX
        if (ext === ".docx") {
            const result = await mammoth.extractRawText({ path: file.path });
            return result.value || "";
        }

        // ✅ PPTX
        if (ext === ".pptx") {
            console.log("📊 Parsing PPTX...");

            let text = "";

            // 🥇 Try pptx2json
            try {
                const rawData = await pptxParser(file.path);

                const slides = Array.isArray(rawData)
                    ? rawData
                    : Object.values(rawData || {});

                text = slides
                    .map(slide =>
                        (slide.texts || [])
                            .map(t => t.text || "")
                            .join(" ")
                    )
                    .join("\n");

                console.log("pptx2json length:", text.length);

            } catch (err) {
                console.warn("⚠️ pptx2json failed:", err.message);
            }

            // 🥈 Fallback → officeparser
            if (!text || text.trim().length < 50) {
                try {
                    console.log("🔁 fallback → officeparser");

                    const data = await officeParser.parseOffice(file.path);

                    text = typeof data === "string"
                        ? data
                        : data?.text || "";

                } catch (err) {
                    console.error("❌ officeparser failed:", err.message);
                }
            }

            if (text && text.trim().length > 50) {
                return text.slice(0, 50000);
            }

            return "";
        }

        return "";

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

app.post("/upload-stream", upload.single("file"), async (req, res) => {
    try {
        const userId = req.headers["x-user-id"];
        const file = req.file;

        if (!file || !userId) {
            return res.status(400).json({ error: "Missing file" });
        }

        console.log("🚀 Upload started");

        const fileKey = `${userId}/${Date.now()}_${file.originalname}`;

        // ✅ STEP 1: STREAM TO R2 (NO RAM SPIKE)
        const fileStream = fs.createReadStream(file.path);

        await r2.send(new PutObjectCommand({
            Bucket: process.env.R2_BUCKET,
            Key: fileKey,
            Body: fileStream,
            ContentType: file.mimetype
        }));

        console.log("✅ Uploaded to R2");

        progressEvents.emit(`update-${userId}`, { step: "Uploading...", progress: 10 });

        // ✅ STEP 2: RESPOND IMMEDIATELY
        res.json({ success: true });

        // =========================
        // 🔥 BACKGROUND PROCESSING
        // =========================
        setImmediate(async () => {
            let tempPath = file.path;

            try {
                console.log("⚙️ Processing started");

                // Extract text
                const text = await extractText({
                path: tempPath,
                originalname: file.originalname
               });

              /*   if (!text || text.length < 50) {
                    console.log("❌ No text extracted");

                 progressEvents.emit(`update-${userId}`, {
                    step: "❌ Failed (No text found)",
                    progress: 100
                });

                  return;
                 }*/

                if (!text || text.length < 50) {

    console.log("❌ No text extracted");

    const fileName = file.originalname || "";
    const ext = fileName.toLowerCase().split(".").pop();

    let message = "❌ Failed (No readable text found)";

    if (ext === "pdf") {
        message = "❌ PDF has no selectable text (scanned file)";
    } else if (ext === "docx") {
        message = "❌ DOCX appears empty or unsupported";
    } else if (ext === "pptx") {
        message = "❌ PPTX parsing failed — try uploading as PDF";
    }

    progressEvents.emit(`update-${userId}`, {
        step: message,
        progress: 100
    });

    return;
                }


                progressEvents.emit(`update-${userId}`, { step: "Extracting...", progress: 30 });

                // Chunk
                const childChunks = recursiveChunk(text, 700, 100);
                const limitedChunks = childChunks.slice(0, 20);

                progressEvents.emit(`update-${userId}`, { step: "Chunking...", progress: 50 });

                // Embed
                const insertBatch = [];

              /*  for (let chunk of limitedChunks) {
                    const vector = await embedText(chunk);

                    insertBatch.push({
                        user_id: userId,
                        filename: file.originalname,
                        content: chunk,
                        embedding: vector
                    });
                } */

                const vectors = await Promise.all(
                limitedChunks.map(chunk => embedText(chunk))
                );

                 vectors.forEach((vector, i) => {
                 insertBatch.push({
                      user_id: userId,
                      filename: file.originalname,
                      content: limitedChunks[i],
                      embedding: vector
                    });
                  });

                progressEvents.emit(`update-${userId}`, { step: "Embedding...", progress: 80 });

                // Save
                await supabase.from("book_chunks").insert(insertBatch);

                await supabase.from("books").insert([
                    { user_id: userId, filename: file.originalname }
                ]);

                progressEvents.emit(`update-${userId}`, { step: "Completed ✅", progress: 100 });

                console.log("✅ Processing complete");

            } catch (err) {
                console.error("❌ Background processing failed:", err);
            } finally {
                // 🧹 DELETE FILE AFTER USE
                if (fs.existsSync(tempPath)) {
                    fs.unlinkSync(tempPath);
                    console.log("🧹 Temp file deleted");
                }
            }
        });

    } catch (err) {
        console.error("❌ Upload error:", err);
        res.status(500).json({ error: "Upload failed" });
    }
});

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

        progressEvents.emit(`update-${userId}`, { step: "Uploading...", progress: 10 });
    });

          response.Body.pipe(writeStream);
 });
        const stats = fs.statSync(tempPath);
         console.log("📦 File size:", stats.size);
        
        // 2. Extract Text
        console.log("STEP 1: Starting upload");
        const text = await extractText({
        path: tempPath,
        originalname: filename
        });
        progressEvents.emit(`update-${userId}`, { step: "Extracting...", progress: 30 });
        console.log("STEP 2: Extracted text length:", text ? text.length : 0);

        // CRITICAL: Stop if extraction failed
        if (!text || text.trim().length < 50) {
            if (fs.existsSync(tempPath)) fs.unlinkSync(tempPath);
            return res.status(422).json({ error: "Could not extract study material from this file." });
        }

        // 3. Chunking
        const parentChunks = recursiveChunk(text, 1500, 200);
        const childChunks = recursiveChunk(text, 700, 100);
        progressEvents.emit(`update-${userId}`, { step: "Chunking...", progress: 50 });

        // 🚀 LIMIT chunks (IMPORTANT)
        const MAX_CHUNKS = 20;
        const limitedChunks = childChunks.slice(0, MAX_CHUNKS);
        console.log("STEP 3: Chunks created:", childChunks.length);

        //if (!documentStore[userId]) documentStore[userId] = {};
       // documentStore[userId][filename] = { parentChunks, childChunks };
     

        console.log("STEP 4: Starting embedding...");

        // Save chunks to Supabase so they survive redeploy
console.log(`⚡️ Embedding ${limitedChunks.length} chunks...`);

const batchSize = 20;
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

    progressEvents.emit(`update-${userId}`, {
    step: "Embedding...",
    progress
});

    console.log(`📊 Progress: ${Math.min(i + batch.length, limitedChunks.length)}/${limitedChunks.length}`);
}

// ONE INSERT (FAST 🚀)
const { error } = await supabase
    .from("book_chunks")
    .insert(insertBatch);

progressEvents.emit(`update-${userId}`, { step: "Saving...", progress: 90 });

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
               // await addToIndex(userId, filename, childChunks);
                console.log(`✅ Indexing complete for: ${filename}`);
            } catch (err) {
                console.error("Indexing failed in background:", err);
            } finally {
                // Safely delete temp file ONLY after extraction is confirmed done
                if (fs.existsSync(tempPath)) fs.unlinkSync(tempPath);
            }
        });

        progressEvents.emit(`update-${userId}`, { step: "Completed ✅", progress: 100 });

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

        
const vectorContext = data ? data.map(row => row.content) : [];
  const combinedContext = vectorContext;

        console.log("Vector search results:", data);
        console.log("Vector results:", vectorContext.length);
        console.log("Hybrid retrieval working:");
//console.log("Keyword results:", keywordResults.length);
console.log("Combined results:", combinedContext.length);

        if (error) {
    console.error("Vector search error:", error);
}
        if (!data || data.length === 0) {
    return res.json({
        explanation: "This topic is not related to your uploaded materials."
    });
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
// 🔥 LIMIT CONTEXT SIZE (CRITICAL FIX)
const limitedChunks = bestChunks.slice(0, 3); // reduce from 5 → 3

const context = limitedChunks
    .map(c => c.slice(0, 800)) // limit each chunk
    .join("\n\n---\n\n");
    

 const prompt = `
You are an elite university-level academic tutor. Your primary objective is to facilitate deep conceptual mastery of the provided study material.

### CORE DIRECTIVES
* **Skip Pleasantries:** Do NOT greet the user. Begin immediately with the academic explanation.
* **Primary Grounding:** Base your response primarily on the provided textbook context.
* **Supplemental Knowledge:** You may use general academic knowledge ONLY to clarify, expand, or simplify concepts already present in the context. Do NOT introduce unrelated topics.
* **Out-of-Scope Handling:** If the context does not contain relevant information, respond EXACTLY with:
"This topic is not related to your uploaded study material."

### PEDAGOGICAL APPROACH
* **Deep Synthesis:** Do not summarize. Break down mechanisms, processes, and cause-effect relationships step by step.
* **Academic Rigor:** Use precise university-level terminology, but explain clearly.
* **Integrated Definitions:** Define complex terms naturally within explanations (no separate definition section).

### FORMATTING REQUIREMENTS
* Use clear headings for major concepts
* Bold key academic terms on first use
* Provide detailed, well-structured explanations (avoid short answers)

Textbook Context:
${context}

Topic:
${topic}

Provide a structured explanation using headings and detailed paragraphs.
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
  
        const { data, error } = await supabase.rpc("match_book_chunks", {
    query_embedding: queryVector,
    match_threshold: 0, 
    match_count: 5,
    p_user_id: userId,
    p_filename: book === "all" ? null : book
});

if (error) {
    console.error("Vector search error:", error);
}

if (!data || data.length === 0) {
    return res.json({
        notes: "This topic is not related to your uploaded materials."
    });
}

const rawChunks = data.map(row => row.content);

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
You are a Distinguished Academic Tutor specializing in synthesizing complex information into high-retention study materials.

### TASK:
Generate comprehensive, university-level study notes based on the provided topic and textbook context.

### CORE OPERATING GUIDELINES:
1. **Primary Source:** Base your notes primarily on the provided textbook context.
2. **Clarification Layer:** You may integrate general academic knowledge ONLY to clarify terms, explain mechanisms, provide analogies, or improve understanding of concepts already present in the context.
3. **No Hallucination:** Do NOT introduce concepts, facts, or topics that are not supported by the provided context.
4. **Relevance Filter:** If the context does not contain relevant information, output EXACTLY:
"This topic is not related to your uploaded study material."
5. **Tone:** Maintain a professional, rigorous, and analytical academic tone.

---

### NOTE STRUCTURE:

# Topic Overview
Provide a high-level synthesis and explain its significance.

# Key Definitions
Define essential terms with nuance and distinctions where relevant.

# Important Concepts
Break down core ideas. Use **bold** for concept names and explain clearly.

# Mechanisms & Functional Processes
Describe step-by-step processes using numbered steps. Emphasize cause-and-effect relationships.

# Synthesis Summary (Bullet Points)
List key takeaways for quick revision.

# Critical Analysis & Exam Tips
* **Common Pitfalls**
* **Potential Exam Questions**
* **Study Strategy (mnemonic or visualization)**

---

Topic:
${topic}

Textbook Context:
${context}
`;
       

        const chat = await groq.chat.completions.create({
            messages:[{role:"user",content:prompt}],
            model:"llama-3.1-8b-instant",
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

const { data, error } = await supabase.rpc("match_book_chunks", {
    query_embedding: queryVector,
    match_threshold: 0,
    match_count: 5,
    p_user_id: userId,
    p_filename: book === "all" ? null : book
});

if (error) {
    console.error("Vector search error:", error);
    return res.json({ quiz: "Search failed. Try again." });
}
if (!data || data.length === 0) {
    return res.json({
        quiz: "This topic is not related to your uploaded materials."
    });
}

const rawChunks = data.map(row => row.content);

// 🔥 Smart rerank
let bestChunks;

if (rawChunks.length > 5) {
    bestChunks = await rerankChunks(topic, rawChunks);
} else {
    bestChunks = rawChunks;
}

// Build context
const context = bestChunks
    .slice(0, 3)
    .map(c => c.slice(0, 800))
    .join("\n\n---\n\n");

        const prompt = `
You are a Senior Academic Examiner specializing in psychometric assessment design. Your goal is to create a high-discrimination, university-level quiz that tests deep conceptual mastery.

### DESIGN PARAMETERS:
1. **Cognitive Level:** Target Bloom’s Taxonomy levels of Application, Analysis, and Evaluation. Avoid simple recall questions.
2. **Primary Source:** Base all questions primarily on the provided textbook context.
3. **Clarification Allowance:** You may use general academic knowledge ONLY to improve clarity or refine question quality, but do NOT introduce unrelated concepts.
4. **No Hallucination:** Do NOT create questions from information not supported by the context.
5. **Relevance Filter:** If the context does not contain sufficient relevant information, output EXACTLY:
"This topic is not related to your uploaded study material."
6. **Competitive Distractors:** Ensure all options are plausible and based on common misconceptions or closely related facts.
7. **Contrastive Explanations:** The explanation MUST justify the correct answer AND explain why at least one strong distractor is incorrect.

---

### QUIZ DATA:
Topic: ${topic}

Textbook Context:
${context}

---

### OUTPUT FORMAT:
[Generate 5 Questions in the following structure]

## Question 1
**Stem:** [The question text, clearly stated]
A) [Option]
B) [Option]
C) [Option]
D) [Option]

**Correct Answer:** [Letter]  
**Academic Rationale:** [A 2–3 sentence explanation focusing on the logical link between the text and the answer, contrasting it against the distractors.]

(repeat for all questions)
`;

      
        
        const chat = await groq.chat.completions.create({
            messages:[{role:"user",content:prompt}],
            model:"llama-3.1-8b-instant",
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

let allChunks = [];

// 🔥 Loop through selected books
for (const book of books) {

    const { data, error } = await supabase.rpc("match_book_chunks", {
        query_embedding: queryVector,
        match_threshold: 0,
        match_count: 3,
        p_user_id: userId,
        p_filename: book === "all" ? null : book
    });

    if (error) {
    console.error("Vector search error:", error);
    return res.json({ answer: "Search failed. Try again." });
    }

    if (data && data.length > 0) {
        allChunks.push(...data.map(row => row.content));
    }
}

if (allChunks.length === 0) {
    return res.json({ answer: "No relevant content found." });
}

        // Extract raw chunks
        const rawChunks = allChunks;

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

/*app.delete("/delete-book/:name", async (req, res) => {
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
});*/

app.delete("/delete-book/:filename", async (req, res) => {
    try {
        const userId = req.headers["x-user-id"];
        const filename = decodeURIComponent(req.params.filename);

        if (!userId || !filename) {
            return res.status(400).json({ error: "Missing data" });
        }

        console.log("🗑️ Deleting:", filename);

        // 🔥 FIND FILE IN R2 (because of timestamp)
        const list = await r2.send(new ListObjectsV2Command({
            Bucket: process.env.R2_BUCKET,
            Prefix: `${userId}/`
        }));

        const target = list.Contents?.find(obj =>
            obj.Key.endsWith(filename)
        );

        if (target) {
            await r2.send(new DeleteObjectCommand({
                Bucket: process.env.R2_BUCKET,
                Key: target.Key
            }));
            console.log("✅ Deleted from R2:", target.Key);
        } else {
            console.warn("⚠️ File not found in R2");
        }

        // 🔥 DELETE FROM SUPABASE (chunks)
        await supabase
            .from("book_chunks")
            .delete()
            .eq("user_id", userId)
            .eq("filename", filename);

        // 🔥 DELETE FROM BOOKS TABLE
        await supabase
            .from("books")
            .delete()
            .eq("user_id", userId)
            .eq("filename", filename);

        console.log("✅ Deleted from DB");

        res.json({ success: true });

    } catch (err) {
        console.error("❌ Delete failed:", err);
        res.status(500).json({ error: "Delete failed" });
    }
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

    const userId = req.query.userId;

progressEvents.on(`update-${userId}`, onProgress);

    req.on("close", () => {
        progressEvents.removeListener(`update-${userId}`, onProgress);
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
