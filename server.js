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


const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY);

const r2 = new S3Client({
  region: "auto",
  endpoint: `https://${process.env.R2_ACCOUNT_ID}.r2.cloudflarestorage.com`,
  credentials: {
    accessKeyId: process.env.R2_ACCESS_KEY_ID,
    secretAccessKey: process.env.R2_SECRET_ACCESS_KEY
  }
});

const app = express();
app.use(cors({ origin: "*", methods: ["GET","POST","DELETE"], allowedHeaders: ["Content-Type", "x-user-id"] }));
app.use(express.json({ limit: "500mb" }));
app.use(express.urlencoded({ extended: true, limit: "500mb" }));

app.use((req, res, next) => {
    req.setTimeout(600000);
    next();
});

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
const TfIdf = natural.TfIdf;

const UPLOAD_DIR = "./uploads";
const CACHE_FILE = "./rag_cache.json";
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR);

/* ---------------------- */
/* MEMORY & INDEX STORES  */
/* ---------------------- */

let documentStore = {};  // { userId: { filename: { parents: [], children: [] } } }
let keywordIndices = {}; // { userId: { filename: tfidf_instance } }
let vectorIndices = {};  // { userId: { filename: [ { text, vector } ] } }


/* ---------------------- */
/* CACHE SYSTEM           */
/* ---------------------- */

function saveCache() {
    fs.writeFileSync(CACHE_FILE, JSON.stringify(documentStore));
}

async function loadCache() {
    if (fs.existsSync(CACHE_FILE)) {
        try {
            documentStore = JSON.parse(fs.readFileSync(CACHE_FILE, "utf8"));
            await rebuildIndex();
        } catch {
            documentStore = {};
        }
    }
}

/* ---------------------- */
/* RAG PIPELINE HELPERS   */
/* ---------------------- */

// STEP 1: Recursive Chunking
function recursiveChunk(text, size = 2000) {
    const paragraphs = text.split(/\n\s*\n/);
    let chunks = [];
    let currentChunk = "";

    for (let p of paragraphs) {
        if ((currentChunk.length + p.length) < size) {
            currentChunk += p + "\n\n";
        } else {
            chunks.push(currentChunk.trim());
            currentChunk = p + "\n\n";
        }
    }
    if (currentChunk) chunks.push(currentChunk.trim());
    return chunks;
}

// Embeddings Tool

async function embedText(text) {
    try {
        const response = await groq.embeddings.create({
            model: "nomic-embed-text-v1",
            input: text.slice(0, 2000)
        });

        return response.data[0].embedding;

    } catch (err) {
        console.error("Groq Embedding Error:", err);

        // return empty vector to avoid crash
        return new Array(1536).fill(0);
    }
}

function cosineSimilarity(a, b) {
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Build Indexes
async function addToIndex(userId, filename, children) {
    if (!keywordIndices[userId]) keywordIndices[userId] = {};
    if (!vectorIndices[userId]) vectorIndices[userId] = {};

    const tfidf = new TfIdf();
    const vectors = [];

    children.forEach(text => tfidf.addDocument(text.toLowerCase()));

   const batchSize = 50;

for (let i = 0; i < children.length; i += batchSize) {

    try {

        const batch = children.slice(i, i + batchSize);

        const batchVectors = await Promise.all(
            batch.map(async text => ({
                text,
                vector: await embedText(text.toLowerCase())
            }))
        );

        batchVectors.forEach(v => tfidf.addDocument(v.text.toLowerCase()));

        vectors.push(...batchVectors);

        console.log(`Indexed ${vectors.length}/${children.length} chunks`);

    } catch (err) {

        console.error(`Batch starting at ${i} failed, skipping..., err`);

    }
}
    keywordIndices[userId][filename] = tfidf;
    vectorIndices[userId][filename] = vectors;
    console.log(`Indices built for ${filename}`);
}

async function rebuildIndex() {
    keywordIndices = {};
    vectorIndices = {};
    for (const userId of Object.keys(documentStore)) {
        for (const filename of Object.keys(documentStore[userId])) {
            const children = documentStore[userId][filename].children || [];
            await addToIndex(userId, filename, children);
        }
    }
    console.log("📚 All indexes rebuilt");
}

/* ---------------------- */
/* STEP 2: HYBRID SEARCH  */
/* ---------------------- */
async function hybridSearch(userId, query, filename, topK = 12) {
    if (!documentStore[userId]) return [];

    let filesToSearch = filename === "all" ? Object.keys(documentStore[userId]) : [filename];
    let combinedResults = {}; 

    const queryVector = await embedText(query.toLowerCase());

    for (const file of filesToSearch) {
        if (!keywordIndices[userId] || !keywordIndices[userId][file]) continue;

        const tfidf = keywordIndices[userId][file];
        const vectors = vectorIndices[userId][file];
        if (!vectors || vectors.length === 0) continue;

        // Semantic Search
        let semanticScores = vectors.map(v => ({
            text: v.text,
            score: cosineSimilarity(queryVector, v.vector)
        })).sort((a,b) => b.score - a.score).slice(0, 15);

        // Keyword Search
        let keywordScores = [];
        tfidf.tfidfs(query.toLowerCase(), (i, measure) => {
            keywordScores.push({ text: vectors[i].text, score: measure });
        });
        keywordScores = keywordScores.sort((a,b) => b.score - a.score).slice(0, 15);

        // Reciprocal Rank Fusion (RRF)
        semanticScores.forEach((m, rank) => {
            const key = `${file}::${m.text}`;
            combinedResults[key] = (combinedResults[m.text] || 0) + (1 / (rank + 60));
        });
        keywordScores.forEach((m, rank) => {
            const key = `${file}::${m.text}`;
            combinedResults[key] = (combinedResults[m.text] || 0) + (1 / (rank + 60));
        });
    }

    return Object.keys(combinedResults)
    .sort((a, b) => combinedResults[b] - combinedResults[a])
    .slice(0, topK)
    .map(key => key.split("::")[1]);
}

/* ---------------------- */
/* STEP 3: RE-RANKER      */
/* ---------------------- */
async function reRankChunks(query, chunks) {
    if (chunks.length <= 3) return chunks;

    const rankingPrompt = `
    User Question: "${query}"
    
    Examine the text snippets. Identify which contain the direct answer or essential context to explain the topic.
    Return ONLY a JSON array of the indices (0, 1, 2...) of the most relevant snippets, ordered best to worst.
    
    Snippets:
    ${chunks.map((c, i) => `[${i}] ${c.substring(0, 300)}...`).join("\n")}`;

    try {
        const response = await groq.chat.completions.create({
            model: "llama-3.3-70b-versatile", 
            messages: [{ role: "user", content: rankingPrompt }],
        });

        let result;

try {
    result = JSON.parse(response.choices[0].message.content);
} catch {
    console.log("Re-rank JSON parse failed");
    return chunks.slice(0,5);
}

    const indices = result.indices || result.relevant_indices || [0,1,2,3];
        return indices.map(idx => chunks[idx]).filter(Boolean).slice(0, 5);
    } catch (err) {
        console.error("Re-ranking fallback triggered");
        return chunks.slice(0, 5);
    }
}

/* ---------------------- */
/* UNIFIED RAG ENGINE     */
/* ---------------------- */
async function retrieveAdvancedContext(userId, topic, book) {
    try {
        // 1. Query Expansion
        const expansion = await groq.chat.completions.create({
            model: "llama-3.3-70b-versatile",
            messages: [{
                role: "system",
                content: "Give me 3 technical keywords or synonyms related to the user's question to help search a textbook. Return only keywords."
            }, { role: "user", content: topic }]
        });
        const enhancedQuery = `${topic} ${expansion.choices[0].message.content}`;

        // 2. Hybrid Search
        const topChunks = await hybridSearch(userId, enhancedQuery, book, 12);
        if (topChunks.length === 0) return "No relevant context found in the library.";

        // 3. Re-Ranking
        const bestChunks = await reRankChunks(topic, topChunks);

        // 4. Parent-Child Mapping (Context Expansion)
        const context = bestChunks.map(childText => {
            let foundParent = childText;
            const files = book === "all" ? Object.keys(documentStore[userId]) : [book];
            for (let f of files) {
                if (!documentStore[userId][f]) continue;
                const parent = documentStore[userId][f].parents.find(p => p.includes(childText));
                if (parent) { foundParent = parent; break; }
            }
            return foundParent;
        });

        return [...new Set(context)].join("\n\n---\n\n");
    } catch (err) {
        console.error("RAG Engine Error:", err);
        return "Error retrieving context.";
    }
}

/* ---------------------- */
/* ROBUST TEXT EXTRACTION */
/* ---------------------- */
async function extractText(file) {
    const ext = path.extname(file.path).toLowerCase();
    try {
        if (ext === ".pdf") {
            const buffer = new Uint8Array(fs.readFileSync(file.path));
            let result = "";
            try {
                if (typeof pdfParse === "function") {
                    const data = await pdfParse(buffer);
                    result = data?.text || "";
                } else if (pdfParse.PDFParse) {
                    const parser = new pdfParse.PDFParse(buffer);
                    const data = await parser.getText();
                    if (typeof data === "string") result = data;
                    else if (data?.text) result = data.text;
                }
            } catch { console.log("pdf-parse fallback..."); }
            if (result && result.trim().length > 100) return result;
        }

        return new Promise(resolve => {
            textract.fromFileWithPath(file.path, { preserveLineBreaks: true }, (err, text) => {
                if (err) return resolve("");
                resolve(text || "");
            });
        });
    } catch { return ""; }
}

const PROFESSOR_SYSTEM_PROMPT = `
You are an expert university professor helping a student study.
Use ONLY the provided textbook context.
- Provide highly detailed explanations.
- Expand concepts thoroughly.
- Explain definitions, mechanisms, and processes.
- Highlight key terms in **bold**.
- Do NOT invent information outside the textbook.
`;

/* ---------------------- */
/* AI ROUTES              */
/* ---------------------- */

app.post("/deep-explain", async (req, res) => {
    try {
        const userId = String(req.headers["x-user-id"] || "");
        if (!userId) return res.status(400).json({ error: "Unauthorized" });

        const { topic, book } = req.body;
        const context = await retrieveAdvancedContext(userId, topic, book);

        const prompt = `Context from library:\n${context}\n\nExplain "${topic}" in very detailed academic depth. Use structured sections and bullet points.`;

        const chat = await groq.chat.completions.create({
            model: "llama-3.3-70b-versatile",
            messages: [
                { role: "system", content: PROFESSOR_SYSTEM_PROMPT },
                { role: "user", content: prompt }
            ]
        });

        res.json({ explanation: chat.choices[0].message.content });
    } catch (error) {
        res.status(500).json({ error: "AI request failed" });
    }
});

app.post("/notes", async (req, res) => {
    try {
        const userId = String(req.headers["x-user-id"] || "");
        const { topic, book } = req.body;
        const context = await retrieveAdvancedContext(userId, topic, book);
        
        const prompt = `Context from library:\n${context}\n\nCreate structured study notes for "${topic}".`;
        const chat = await groq.chat.completions.create({
            model: "llama-3.3-70b-versatile",
            messages: [{ role: "system", content: PROFESSOR_SYSTEM_PROMPT }, { role: "user", content: prompt }]
        });
        res.json({ notes: chat.choices[0].message.content });
    } catch (error) { res.status(500).json({ error: "Failed to generate notes" }); }
});

app.post("/quiz", async (req, res) => {
    try {
        const userId = String(req.headers["x-user-id"] || "");
        const { topic, book } = req.body;
        const context = await retrieveAdvancedContext(userId, topic, book);
        
        const prompt = `Context from library:\n${context}\n\nCreate a difficult 5-question MCQ quiz about "${topic}".`;
        const chat = await groq.chat.completions.create({
            model: "llama-3.3-70b-versatile",
            messages: [{ role: "system", content: PROFESSOR_SYSTEM_PROMPT }, { role: "user", content: prompt }]
        });
        res.json({ quiz: chat.choices[0].message.content });
    } catch (error) { res.status(500).json({ error: "Failed to generate quiz" }); }
});

/* ---------------------- */
/* CLOUD UPLOAD ROUTES    */
/* ---------------------- */

app.post("/generate-upload-url", async (req, res) => {
  const userId = req.headers["x-user-id"];
  const { filename } = req.body;
  if (!userId || !filename) return res.status(400).json({ error: "Missing user or filename" });

  const key = `${userId}/${Date.now()}_${filename}`;
  const command = new PutObjectCommand({ Bucket: process.env.R2_BUCKET, Key: key });

  try {
    const uploadUrl = await getSignedUrl(r2, command, { expiresIn: 600 });
    res.json({ uploadUrl, fileKey: key });
  } catch (err) { res.status(500).json({ error: "Upload URL generation failed" }); }
});

app.post("/upload", async (req, res) => {
  try {
    const userId = req.headers["x-user-id"];
    const { fileKey, filename } = req.body;
    if (!userId || !fileKey) return res.status(400).json({ error: "Missing file info" });

    // 1. Secure Download from R2
    const command = new GetObjectCommand({ Bucket: process.env.R2_BUCKET, Key: fileKey });
    const response = await r2.send(command);
    const tempPath = path.join(UPLOAD_DIR, Date.now() + "-" + filename);

    const writeStream = fs.createWriteStream(tempPath);

    await new Promise((resolve, reject) => {
    response.Body.pipe(writeStream);
    response.Body.on("error", reject);
    writeStream.on("finish", resolve);
    });
    // 2. Extract Text
    let text = await extractText({ path: tempPath });
   if (!text || text.length < 50) {
    console.log("⚠️ Low text extraction, continuing anyway...");
    text = text || "";
}

    // 3. Parent-Child Chunking
    const parentChunks = recursiveChunk(text, 2000);
    const childChunks = recursiveChunk(text, 500);

    // 4. Save to Database
    await supabase.from("books").insert([{ user_id: userId, filename: filename }]);

    // 5. Build Memory Stores
    documentStore[userId] = documentStore[userId] || {};
    documentStore[userId][filename] = { parents: parentChunks, children: childChunks };
    saveCache();

    // 6. Index Asynchronously
    setImmediate(async () => {
        try {
            await addToIndex(userId, filename, childChunks);
        } catch (err) { console.error("Index rebuild failed:", err); }
    });

    fs.unlinkSync(tempPath);
    res.json({ name: filename, success: true });

  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Upload failed." });
  }
});

/* ---------------------- */
/* BASIC API ROUTES       */
/* ---------------------- */

app.get("/books", async (req, res) => {
    try {
        const userId = req.headers["x-user-id"];
        const { data, error } = await supabase.from("books").select("filename").eq("user_id", userId);
        if (error) return res.status(500).json({ error: "Failed to load books" });
        res.json(data.map(book => ({ name: book.filename })));
    } catch (err) { res.status(500).json({ error: "Server error" }); }
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

app.get("/health", (req, res) => res.json({ status: "alive" }));

const PORT = process.env.PORT || 5000;
app.listen(PORT, async () => {
    await loadCache();
    console.log("🚀 Advanced RAG Server running on port " + PORT);
});