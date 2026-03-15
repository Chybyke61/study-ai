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
const { pipeline } = require("@xenova/transformers");

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

async function extractText(file) {
    const ext = path.extname(file.path).toLowerCase();
    
    try {
        if (ext === ".pdf") {
            const buffer = fs.readFileSync(file.path);
            
            // Attempt 1: pdf-parse
            try {
                const data = await pdfParse(buffer);
                if (data && data.text && data.text.trim().length > 10) {
                    return data.text;
                }
            } catch (pdfErr) {
                console.warn(`⚠️ pdf-parse range error on ${path.basename(file.path)}. Switching to fallback...`);
            }
        }

        // Attempt 2: Textract (Fallback)
        return new Promise((resolve) => {
            textract.fromFileWithPath(file.path, { preserveLineBreaks: true }, (err, text) => {
                if (err) {
                    console.error("❌ All extraction methods failed:", err);
                    return resolve("");
                }
                resolve(text || "");
            });
        });

    } catch (globalErr) {
        console.error("🔥 Critical extraction crash:", globalErr);
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

    children.forEach(chunk => tfidf.addDocument(chunk.toLowerCase()));

    for (let i = 0; i < children.length; i += batchSize) {
        try {
            const batch = children
                .slice(i, i + batchSize)
                .filter(t => t && t.trim().length > 20);

            const batchVectors = await Promise.all(
                batch.map(async text => ({
                    text,
                    vector: await embedText(String(text).toLowerCase())
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
            response.Body.pipe(writeStream);
            response.Body.on("error", (err) => { writeStream.close(); reject(err); });
            writeStream.on("finish", resolve);
        });

        // 2. Extract Text
        const text = await extractText({ path: tempPath });

        // CRITICAL: Stop if extraction failed
        if (!text || text.trim().length < 50) {
            if (fs.existsSync(tempPath)) fs.unlinkSync(tempPath);
            return res.status(422).json({ error: "Could not extract study material from this file." });
        }

        // 3. Chunking
        const parentChunks = recursiveChunk(text, 1500, 200);
        const childChunks = recursiveChunk(text, 400, 50);

        if (!documentStore[userId]) documentStore[userId] = {};
        documentStore[userId][filename] = { parentChunks, childChunks };

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

        let booksToSearch = [];

        if (book === "all") {
            booksToSearch = Object.keys(vectorIndices[userId] || {});
        } else {
            booksToSearch = [book];
        }

        const queryVector = await embedText(topic.toLowerCase());

        let results = [];

        for (const b of booksToSearch) {

            const vectors = vectorIndices[userId]?.[b];
            const chunks = documentStore[userId]?.[b]?.childChunks;

            if (!vectors || !chunks) continue;

            vectors.forEach((vecObj, i) => {
                const score = cosineSimilarity(queryVector, vecObj.vector);
            results.push({ score, text: chunks[i] });
            });
        }

        if (!vectors || !chunks) {
            return res.json({ explanation: "No context found for this book." });
        }

        vectors.forEach((vecObj, i) => {
            const score = cosineSimilarity(queryVector, vecObj.vector);
            results.push({ score, text: chunks[i] });
        });

        results.sort((a,b)=>b.score-a.score);

        const context = results.slice(0,5).map(r=>r.text).join("\n\n---\n\n");

        const prompt = `
You are an expert professor.
Explain the topic clearly using the textbook excerpts.

Context:
${context}

Topic:
${topic}
`;

        const chat = await groq.chat.completions.create({
            messages:[{role:"user",content:prompt}],
            model:"llama-3.3-70b-versatile"
        });

        res.json({ explanation: chat.choices[0].message.content });

    } catch(err){
        console.error(err);
        res.status(500).json({ error:"Explain failed" });
    }
});

app.post("/notes", async (req, res) => {
    try {
        const { topic, book } = req.body;
        const userId = req.headers["x-user-id"];

        const queryVector = await embedText(topic.toLowerCase());

        let results = [];

        const vectors = vectorIndices[userId]?.[book];
        const chunks = documentStore[userId]?.[book]?.childChunks;

        vectors.forEach((vecObj,i)=>{
            const score = cosineSimilarity(queryVector, vecObj.vector);
            results.push({score,text:chunks[i]});
        });

        results.sort((a,b)=>b.score-a.score);

        const context = results.slice(0,5).map(r=>r.text).join("\n\n");

        const prompt = `
Create structured study notes from the textbook.

Topic: ${topic}

Context:
${context}
`;

        const chat = await groq.chat.completions.create({
            messages:[{role:"user",content:prompt}],
            model:"llama-3.3-70b-versatile"
        });

        res.json({ notes: chat.choices[0].message.content });

    } catch(err){
        res.status(500).json({ error:"Notes failed" });
    }
});

app.post("/quiz", async (req, res) => {
    try {
        const { topic, book } = req.body;
        const userId = req.headers["x-user-id"];

        const queryVector = await embedText(topic.toLowerCase());

        let results = [];

        const vectors = vectorIndices[userId]?.[book];
        const chunks = documentStore[userId]?.[book]?.childChunks;

        vectors.forEach((vecObj,i)=>{
            const score = cosineSimilarity(queryVector, vecObj.vector);
            results.push({score,text:chunks[i]});
        });

        results.sort((a,b)=>b.score-a.score);

        const context = results.slice(0,5).map(r=>r.text).join("\n\n");

        const prompt = ` 
Create a difficult 5-question multiple choice quiz about:

${topic}

Use the textbook context below.

${context}
`;

        const chat = await groq.chat.completions.create({
            messages:[{role:"user",content:prompt}],
            model:"llama-3.3-70b-versatile"
        });

        res.json({ quiz: chat.choices[0].message.content });

    } catch(err){
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
        const topMatches = allResults.slice(0, 5).map(r => r.text).join("\n\n---\n\n");

        const prompt = `You are a helpful study assistant. Use the textbook excerpts to answer the question.\n\nContext:\n${topMatches}\n\nQuestion: ${query}`;

        const chatCompletion = await groq.chat.completions.create({
            messages: [{ role: "user", content: prompt }],
            model: "llama-3.3-70b-versatile",
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
app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));