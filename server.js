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
const { GoogleGenAI } = require("@google/genai");

const genAI = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY
});
const natural = require("natural");
const { createClient } = require('@supabase/supabase-js');
const { S3Client, PutObjectCommand, GetObjectCommand, DeleteObjectCommand, ListObjectsV2Command } = require("@aws-sdk/client-s3");
const { getSignedUrl } = require("@aws-sdk/s3-request-presigner");
const { pipeline, max } = require("@xenova/transformers");
const e = require("express");
const mammoth = require("mammoth");
const officeParser = require("officeparser");
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
//const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
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
const userQuizAttempts = {};
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

/*async function geminiGenerate(prompt) {
    try {
        const result = await genAI.models.generateContent({
  model: "gemini-3.1-flash-lite-preview", // Lighter model for better availability
  contents: prompt
});

return result.text;
    } catch (err) {
        console.error("Gemini failed:", err);
        throw err;
    }
} */

async function geminiGenerate(prompt, retries = 3) {
    for (let i = 0; i < retries; i++) {
        try {
            const result = await genAI.models.generateContent({
                model: "gemini-3.1-flash-lite-preview", // Lighter model for better availability
                contents: [{
                    role: "user",
                    parts: [{ text: prompt }]
                }]
            });

            return result.text;

        } catch (err) {
            console.warn(`Gemini attempt ${i + 1} failed`);

            if (i === retries - 1) throw err;

            await new Promise(r => setTimeout(r, 2000)); // wait 2s
        }
    }
}


async function safeGenerate(prompt) {
    try {
        const chat = await groq.chat.completions.create({
            messages: [{ role: "user", content: prompt }],
            model: "llama-3.1-8b-instant"
        });

        return chat.choices[0].message.content;

    } catch (err) {
        console.warn("⚠️ Groq failed → Gemini fallback");
        return await geminiGenerate(prompt);
    }
}

async function geminiOCR(filePath, mimeType = "image/png") {
    try {
        console.log("🔍 Gemini OCR running...");
        const fileBuffer = fs.readFileSync(filePath);

        const result = await genAI.models.generateContent({
  model: "gemini-2.5-flash",
  contents: [
    {
      role: "user",
      parts: [
        {
          inlineData: {
            data: fileBuffer.toString("base64"),
            mimeType
          }
        },
        {
          text: "Extract all readable text from this document. Return only clean text."
        }
      ]
    }
  ]
});

const text = result.text || "";

        console.log("✅ Gemini OCR done:", text.length);

        return text;

    } catch (err) {
        console.error("❌ Gemini OCR failed:", err);
        return "";
    }
}


function isBadText(text) {
    if (!text || text.length < 200) return true;

    const clean = text.replace(/\s+/g, " ").trim();
    const words = clean.split(" ");

    if (words.length < 30) return true;

    const charCount = clean.length;

    const letters = (clean.match(/[a-zA-Z]/g) || []).length;
    const letterRatio = letters / charCount;
    if (letterRatio < 0.6) return true;

    const weirdChars = (clean.match(/[^a-zA-Z0-9 .,?!'"()\-]/g) || []).length;
    if ((weirdChars / charCount) > 0.1) return true;

    const realWords = words.filter(w => /^[a-zA-Z]{3,}$/.test(w)).length;
    const realWordRatio = realWords / words.length;
    if (realWordRatio < 0.4) return true;

    // 🔥 NEW: catch fake repeated nonsense words
    const uniqueWords = new Set(words.map(w => w.toLowerCase()));
    const diversityRatio = uniqueWords.size / words.length;

    if (diversityRatio < 0.3) return true; // 🚨 KEY ADDITION

    return false;
}


async function extractText(file) {
    const fileName = file.originalname || file.filename || file.path || "";
const ext = path.extname(fileName).toLowerCase();
    try {
        //==========
        // ✅ PDF
        //==========
        if (ext === ".pdf") {
        console.log("📄 Parsing PDF...");

        const buffer = fs.readFileSync(file.path);
        const data = await pdfParse(buffer);

        const text = data.text || "";

        const cleanedText = text
    .replace(/[^\x00-\x7F]/g, " ")  // remove weird symbols
    .replace(/\s+/g, " ")           // normalize spaces
    .trim();

        console.log("PDF text length:", cleanedText.length);
        console.log("Preview:", cleanedText.slice(0, 100));
            
        // 🔥 ACCEPT ANY REAL TEXT
      const bad = isBadText(cleanedText);

console.log("🧠 BadText check:", bad);
console.log("📊 Stats:", {
    length: cleanedText.length,
    words: cleanedText.split(" ").length
});

if (!bad) {
    console.log("✅ Clean PDF detected");
    return cleanedText;
}

// 🔥 OCR FALLBACK HERE
console.warn("⚠️ Scanned PDF → using Gemini OCR");

const ocrText = await geminiOCR(file.path, "application/pdf");

if (ocrText && ocrText.trim().length > 30) {
    console.log("✅ OCR used");
    return ocrText.replace(/\s+/g, " ").trim();
}

console.warn("❌ OCR failed or too short");
return "";
        } 


        // ======================
// ✅ PPTX SUPPORT (SAFE)
// ======================

if (ext === ".pptx") {
    console.log("📊 PPT detected → using officeparser v6 (Safe Extension Mode)");

    // 1. Create a temporary path that DEFINITELY ends in .pptx
    // OfficeParser needs this to identify the file type
    const tempPptxPath = path.resolve(file.path + ".pptx");
    
    try {
        // 2. Rename/Copy the file so it has the .pptx extension
        fs.copyFileSync(path.resolve(file.path), tempPptxPath);
        console.log("📂 Processing file at:", tempPptxPath);

        // 3. Parse the file with the proper extension
        const rawData = await officeParser.parseOffice(tempPptxPath);
        
        let extractedText = "";
        if (rawData && typeof rawData === "object") {
            extractedText = typeof rawData.toText === "function" ? rawData.toText() : JSON.stringify(rawData);
        } else {
            extractedText = String(rawData);
        }

        // 4. Clean up the temp .pptx file immediately
        if (fs.existsSync(tempPptxPath)) fs.unlinkSync(tempPptxPath);

        const finalOutput = extractedText
            .split('\n')
            .map(line => line.trim())
            .filter(line => line.length > 0)
            .join('\n');

        if (finalOutput.length > 50) {
            console.log("✅ PPT extraction successful");
            return finalOutput;
        }

    } catch (error) {
        console.error("❌ PPT extraction failed:", error.message);
        // Clean up temp file if error occurs
        if (fs.existsSync(tempPptxPath)) fs.unlinkSync(tempPptxPath);
    }
    return "";
}


        
 // =====================
// ✅ DOCX FIX
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

    // 🔥 SKIP RERANK IF SMALL DATA
if (!chunks || chunks.length <= 5) {
    return chunks;
}
    
    try {
        const limitedChunks = chunks.slice(0, 6); // 🔥 limit for free tier

        const prompt = `
Select the 5 most relevant chunks for the query.

Query:
${query}

Chunks:
${limitedChunks.map((c, i) => `[${i}] ${c}`).join("\n\n")}

Return ONLY numbers like: 2,5,1,3,0
`;

        let text;

try {
    const response = await groq.chat.completions.create({
        model: "llama-3.1-8b-instant",
        messages: [{ role: "user", content: prompt }],
        temperature: 0
    });

    text = response.choices[0].message.content;

} catch (err) {
    console.warn("⚠️ Rerank fallback → Gemini");
    text = await geminiGenerate(prompt);
}

        const indices = text.match(/\d+/g)?.map(Number) || [];

        return indices
            .map(i => limitedChunks[i])
            .filter(Boolean);

    } catch (err) {
        console.error("Rerank failed:", err);
        return chunks.slice(0, 5); // fallback
    }
}


function isComplexQuery(query) {
    if (!query) return false;

    const q = query.toLowerCase();

    return (
        q.split(" ").length > 12 || // long query
        q.includes("compare") ||
        q.includes("difference") ||
        q.includes("mechanism") ||
        q.includes("process") ||
        q.includes("why") ||
        q.includes("how")
    );
}

async function rewriteQuery(topic) {
    try {
        const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${process.env.GROQ_API_KEY}`,
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                model: "llama-3.1-8b-instant",
                messages: [
                    {
                        role: "system",
                        content: `
Rewrite the user's topic into a detailed academic search query.

Make it:
- specific
- concept-rich
- include key terms

Return ONLY the improved query.
`
                    },
                    {
                        role: "user",
                        content: topic
                    }
                ]
            })
        });

        const data = await res.json();

const content = data?.choices?.[0]?.message?.content;

if (!content) {
    console.warn("Rewrite failed, using original query");
    return topic;
}

const cleaned = content.trim();

// 🔥 BLOCK BAD AI RESPONSES
if (
    cleaned.toLowerCase().includes("there is no") ||
    cleaned.toLowerCase().includes("i need you to") ||
    cleaned.length > 200
) {
    console.warn("⚠️ Bad rewrite detected, fallback used");
    return topic;
}

return cleaned;

        
    } catch (err) {
        console.error("Rewrite failed:", err);
        return topic;
    }
}


/**
 * Analyzes user intent using Llama 3 via Groq.
 * Improvements: Added timeout, HTTP error handling, and robust JSON extraction.
 */
async function analyzeIntent(query) {
    const API_URL = "https://api.groq.com/openai/v1/chat/completions";
    const MODEL = "llama-3.1-8b-instant";
    
    // Default fallback object
    const fallback = { intent: "explain", query: query };

    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 8000); // 8s timeout

        const response = await fetch(API_URL, {
            method: "POST",
            signal: controller.signal,
            headers: {
                "Authorization": `Bearer ${process.env.GROQ_API_KEY}`,
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                model: MODEL,
                messages: [
                    {
                        role: "system",

                        content: `
You are an AI that classifies academic user intent.

Return ONLY JSON:

{
  "intent": "explain | summary | notes | quiz | abstract",
  "query": "improved academic query"
}

Rules:
- "summary" → user wants overview of entire material
- "abstract" → user asks for idea, concept, intuition, big picture
- "explain" → detailed explanation of a topic
- "notes" → key points / bullet summary
- "quiz" → questions/testing

IMPORTANT:
- Rewrite vague queries into clear academic search queries
- If user says "this book" → assume full document context
- Keep query meaningful for retrieval
`
                        
                    },
                    { role: "user", content: query }
                ],
                temperature: 0.1, // Lower temperature for more consistent JSON
                response_format: { type: "json_object" } // Groq supports JSON mode
            })
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(`Groq API error: ${response.status} - ${JSON.stringify(errorData)}`);
        }

        const data = await response.json();
        const content = data.choices?.[0]?.message?.content;

        if (!content) return fallback;

        // 🔥 ROBUST PARSING: Handles potential markdown backticks or whitespace

   const cleanedContent = content.replace(/```json|```/g, "").trim();

let parsed = {};

try {
    parsed = JSON.parse(cleanedContent);
} catch (parseError) {
    console.warn("JSON Parse failed. Content was:", cleanedContent);
    parsed = {};
}
        
  if (query.split(" ").length < 4 && !parsed.query) {
    parsed.query = query + " detailed explanation with examples";
  }
        
            // 🔥 Manual boost for abstract questions
const lower = query.toLowerCase();

if (
    lower.includes("idea") ||
    lower.includes("concept") ||
    lower.includes("meaning") ||
    lower.includes("intuition") ||
    lower.includes("what is the idea behind")
) {
    parsed.intent = "abstract";
}
            

            let finalQuery = parsed.query || query;

// 🔥 Fix useless queries like "this", "this book"
if (
    finalQuery.length < 8 ||
    finalQuery.toLowerCase().trim() === "this"
) {
    finalQuery = query + " detailed academic explanation with key concepts";
}

 // 🔥 Add domain context
parsed.query += " academic explanation key concepts examples";

return {
    intent: parsed.intent || fallback.intent,
    query: finalQuery
};
        

    } catch (err) {
        if (err.name === 'AbortError') {
            console.error("Intent analysis timed out");
        } else {
            console.error("Intent analysis error:", err.message);
        }
        return fallback;
    }
}

// SMART QUIZ QUERY ENGINE
function smartQuizQuery(query, context = "") {
    const lower = query.toLowerCase().trim();

    // Extremely vague input
    if (lower.length < 4) {
        return "Generate a quiz based on the key concepts in the uploaded material";
    }

    // vague words
    if (["this", "that", "it", "something", "stuff"].includes(lower)) {
        return "Generate a quiz from the main topics and key concepts in the uploaded material";
    }

    // vague phrases
    if (lower.includes("this book") || lower.includes("this topic")) {
        return "Generate a quiz from the important concepts, definitions, and applications in the uploaded material";
    }

    // "quiz this"
    if (lower.includes("quiz")) {
        return query + " with important concepts, definitions, and exam-style questions";
    }

    // ✅ normal query → enhance it
    return query + " with key concepts, applications, and exam-style questions";
}

//CBT FALL BACK
async function generateCBTContent(prompt) {
    const MAX_LENGTH = 12000;
    const trimmedPrompt = prompt.length > MAX_LENGTH 
        ? prompt.slice(0, MAX_LENGTH) 
        : prompt;

    try {
        console.log("🔥 CBT → Groq");

        const chat = await groq.chat.completions.create({
            messages: [{ role: "user", content: trimmedPrompt }],
            model: "llama-3.1-8b-instant",
            temperature: 0.3,
            max_tokens: 2500
        });

        return chat.choices[0].message.content;

    } catch (err) {
        console.warn("⚠️ CBT Groq failed → Gemini fallback");

        try {
            const result = await Promise.race([
                geminiGenerate(trimmedPrompt),
                new Promise((_, reject) =>
                    setTimeout(() => reject(new Error("Gemini timeout")), 8000)
                )
            ]);

            return result;

        } catch (err2) {
            console.error("❌ CBT Gemini failed → fallback questions");

            return generateBasicCBT();
        }
    }
}

function generateBasicCBT() {
    return `
Question:
What is the main idea of the text?

A. First concept
B. Second concept
C. Third concept
D. Fourth concept

Answer: A
Explanation: Based on available content.
`;
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

                 if (!text || text.length < 50) {
                    console.log("❌ No text extracted");

                 progressEvents.emit(`update-${userId}`, {
                    step: "❌ Failed (No text found)",
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
        const text = await extractText({ path: tempPath });
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

        // 🧠 Step 1: Understand intent
const analysis = await analyzeIntent(topic);

// 🚨 SAFE SUMMARY MODE (LOW RAM, BYPASS SEARCH)
if (analysis.intent === "summary") {

    console.log("📘 Summary mode activated");

    const { data: allChunks } = await supabase
        .from("book_chunks")
        .select("content")
        .eq("user_id", userId)
        .eq("filename", book === "all" ? undefined : book);

    if (!allChunks || allChunks.length === 0) {
        return res.json({
            explanation: "No content found to summarize."
        });
    }

    // STRICT LIMITS 
    const MAX_CHUNKS = 8;

    const context = allChunks
        .slice(0, MAX_CHUNKS)
        .map(row => row.content.slice(0, 400)) // trim each chunk
        .join("\n\n---\n\n");

    console.log("🧠 Summary context size:", context.length);

    /*const prompt = `
You are a university-level tutor.

Summarize the following material into:
- Key ideas
- Important concepts
- Clear structure

Material:
${context}
`;*/

    const prompt = `
You are a university-level academic tutor.

Your task is to generate a DETAILED and WELL-STRUCTURED summary of the material.

### REQUIREMENTS:
- Do NOT be brief
- Cover ALL important concepts
- Expand explanations clearly
- Maintain academic depth

### FORMAT:

# Overview
Explain the main idea of the material in detail.

# Key Concepts
Explain each important concept clearly with explanation.

# Important Details
Include supporting explanations, examples, or processes.

# Summary Points
Provide bullet points for revision.

### RULES:
- Stay faithful to the material
- Do NOT skip important sections
- Do NOT shorten excessively

Material:
${context}
`;

    const output = await safeGenerate(prompt);

    return res.json({ explanation: output });
}

const intent = analysis.intent;
const improvedQuery = analysis.query;
        
// 🔥 Expand query for better understanding
let expandedQuery = await rewriteQuery(improvedQuery);

// 🔥 PREVENT GENERIC / USELESS QUERIES
if (
    expandedQuery.toLowerCase().includes("provided text") ||
    expandedQuery.toLowerCase().includes("academic summary")
) {
    console.warn("⚠️ Generic query detected, reverting");
    expandedQuery = topic;
}

let searchQuery = expandedQuery;

// 🔥 Boost complex queries
if (isComplexQuery(topic)) {
    searchQuery += " detailed mechanism explanation examples";
}

// 🔥 SMART MODES
if (intent === "summary") {
    searchQuery = topic + " main ideas key concepts important points summary";
}

if (intent === "abstract") {
    searchQuery = topic + " fundamental concept intuition explanation";
}

console.log("Expanded:", expandedQuery);

console.log("Intent:", intent);
console.log("Improved:", improvedQuery);

// 🧠 Step 2: Embed smarter query
const queryVector = await embedText(searchQuery.toLowerCase());
        
            const { data, error } = await 
            supabase.rpc("match_book_chunks", {
            query_embedding: queryVector,
            match_threshold: 0.2,
            match_count: 8,
            p_user_id: userId,
            p_filename: book === "all" ? null : book
        });


        // 🚨 HARD RELEVANCE CHECK
const MIN_SIMILARITY = 0.3;

const validMatches = data.filter(row => row.similarity >= MIN_SIMILARITY);

if (!validMatches || validMatches.length === 0) {
    return res.json({
        explanation: "Not found in your uploaded documents"
    });
}

        
// 🔥 Filter weak matches (simulate scoreThreshold)
let vectorContext = [];

if (data && data.length > 0) {

    // ✅ Sort by best similarity first
    const sorted = data.sort((a, b) => b.similarity - a.similarity);

    // ✅ Always take top results
    let limit = 6;

if (intent === "summary") limit = 15;
if (intent === "abstract") limit = 12;
if (intent === "notes") limit = 10;
if (intent === "quiz") limit = 8;

const topK = sorted.slice(0, limit);

    // ✅ Only filter if strong matches exist
    let threshold = 0.25;

if (intent === "summary") threshold = 0.15;
if (intent === "abstract") threshold = 0.1;
if (intent === "notes") threshold = 0.2;
if (intent === "explain") threshold = 0.25;

const strongMatches = topK.filter(row => row.similarity > threshold);

    const finalChunks = strongMatches.length > 0 ? strongMatches : topK;

    vectorContext = finalChunks.map(row => row.content);
}
        
        
  const combinedContext = vectorContext;
        
   // 🔥 Boost top chunk priority
if (combinedContext.length > 0) {
    combinedContext.unshift(combinedContext[0]);
}
        
        console.log("Vector search results:", data);
        console.log("Vector results:", vectorContext.length);
        console.log("Hybrid retrieval working:");
//console.log("Keyword results:", keywordResults.length);
console.log("Combined results:", combinedContext.length);

        if (error) {
    console.error("Vector search error:", error);
        }

if (!data || data.length === 0) {
    console.warn("⚠️ No vector matches, retrying with raw topic");

    const fallbackVector = await embedText(topic.toLowerCase());

    const fallbackSearch = await supabase.rpc("match_book_chunks", {
        query_embedding: fallbackVector,
        match_threshold: 0.1,
        match_count: 5,
        p_user_id: userId,
        p_filename: book === "all" ? null : book
    });

    if (!fallbackSearch.data || fallbackSearch.data.length === 0) {
        return res.json({
            explanation: "This topic is not related to your uploaded materials."
        });
    }

    data = fallbackSearch.data;
}

        
// Extract raw chunks
const rawChunks = combinedContext;

// 🔥 Smart rerank (only when needed)
let bestChunks;

if (
    rawChunks.length > 8 &&
    (isComplexQuery(topic) || intent === "summary" || intent === "abstract")
) {
    console.log("🧠 Smart rerank activated");
    bestChunks = await rerankChunks(expandedQuery, rawChunks);
} else {
    bestChunks = rawChunks;
}
        
// Build final context
// 🔥 LIMIT CONTEXT SIZE (CRITICAL FIX)
let limit = 6;

if (intent === "summary") limit = 10;
if (intent === "abstract") limit = 8;

const limitedChunks = bestChunks
    .filter(c => c && c.length > 200)
    .slice(0, limit);

const context = limitedChunks
    .filter(c => c && c.length > 200)
    .join("\n\n---\n\n");

        console.log("🧠 FINAL CONTEXT:\n", context);
        
   let modeInstruction = "";

if (intent === "summary") {
    modeInstruction = "Provide a structured summary with headings and key points.";
}

if (intent === "abstract") {
    modeInstruction = "Start with intuition and big picture before technical details.";
}

if (intent === "notes") {
    modeInstruction = "Provide bullet-point notes with key concepts.";
}

if (intent === "quiz") {
    modeInstruction = "Generate challenging conceptual questions.";
} 

 const prompt = `
You are an elite university-level academic tutor. Your primary objective is to facilitate deep conceptual mastery of the provided study material.

### SPECIAL INSTRUCTION:
${modeInstruction}

### CORE DIRECTIVES
- Skip Pleasantries: Do NOT greet the user. Begin immediately with the academic explanation.
- Primary Grounding: Base your response primarily on the provided textbook context.
- STRICT RULE: You are ONLY allowed to use the provided context.
- Do NOT use any external or prior knowledge.
- If the answer is not explicitly found in the context, respond ONLY with:
  "Not found in your uploaded documents"
- Use wording and phrasing similar to the textbook.
- Stay close to the original text.
- You must answer ONLY using the provided study material.

### PEDAGOGICAL APPROACH
- Deep Synthesis: Do not summarize. Break down mechanisms, processes, and cause-effect relationships step by step.
- Academic Rigor: Use precise university-level terminology, but explain clearly.
- Integrated Definitions: Define complex terms naturally within explanations (no separate definition section).
- Explain thoroughly 

### SPECIAL MODES:
- If intent = "summary": give a concise structured overview
- If intent = "abstract": explain concept at high-level (big picture, intuition first)
- If intent = "explain": deep technical breakdown

### THINKING STRATEGY:
1. Identify key concepts in the context
2. Explain relationships between them
3. Build explanation step-by-step
4. Conclude with integrated understanding

### RULES:
- Use ONLY provided context
- Do NOT hallucinate
- If information is missing, explicitly say:
  "Not found in your uploaded documents"
- Do not guess or invent facts



### FORMATTING REQUIREMENTS
- Use clear headings for major concepts
- Bold key academic terms on first use
- Provide detailed, well-structured explanations (avoid short answers)

Textbook Context:
${context}

Topic:
${topic}

Provide a structured explanation using headings and detailed paragraphs.
`;

        let output;

try {
    const chat = await groq.chat.completions.create({
        messages: [{ role:"user", content:prompt }],
        model: "llama-3.1-8b-instant",
        temperature: 0.2,
        max_tokens: 1500
    });

    output = chat.choices[0].message.content;

} catch (err) {

    console.warn("⚠️ Groq failed → switching to Gemini");

// 🔥 ALWAYS fallback (safe mode)
try {
    output = await geminiGenerate(prompt);
} catch (gemErr) {
    console.error("❌ Gemini also failed:", gemErr);
    throw new Error("Both AI providers failed");
}
}

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
  
        let { data, error } = await supabase.rpc("match_book_chunks", {
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
        // 🔥 SMART QUIZ QUERY (DO NOT BLOCK USER)
        const smartQuery = smartQuizQuery(topic);
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

      
        const queryVector = await embedText(smartQuery.toLowerCase());

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
    bestChunks = await rerankChunks(smartQuery, rawChunks);
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
Topic: ${smartQuery}

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

        let output = chat.choices[0].message.content;
        
        // 🔥 FIX BROKEN QUIZ FORMAT
if (!output || !output.includes("## Question 1")) {
    console.warn("⚠️ Quiz format broken → fallback");

    try {
        const fallback = await geminiGenerate(prompt);

        if (fallback && fallback.includes("## Question 1")) {
            output = fallback;
        }
    } catch (err) {
        console.error("Fallback failed:", err);
    }
}
        

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

// ==============================
// 🧠 FLASHCARDS GENERATOR
// ==============================
app.post("/flashcards", async (req, res) => {
    try {
        const { topic, level, book } = req.body;
        const userId = req.headers["x-user-id"];

        if (!topic) {
            return res.status(400).json({ error: "Topic required" });
        }

        // 🔍 GET RELEVANT CHUNKS (same system as quiz)
        const queryVector = await embedText(topic.toLowerCase());

        const { data, error } = await supabase.rpc("match_book_chunks", {
            query_embedding: queryVector,
            match_threshold: 0,
            match_count: 5,
            p_user_id: userId,
            p_filename: book === "all" ? null : book
        });

        if (error || !data || data.length === 0) {
            return res.json({
                flashcards: "No relevant study material found."
            });
        }

        // 🔥 LIMIT CONTEXT (VERY IMPORTANT FOR RAM)
        const context = data
            .slice(0, 3)
            .map(row => row.content.slice(0, 500))
            .join("\n\n");

        const prompt = `
You are a study assistant.

Create 5 flashcards from the material below.

Format strictly as:
Q: ...
A: ...

Material:
${context}
`;

        const chat = await groq.chat.completions.create({
            messages: [{ role: "user", content: prompt }],
            model: "llama-3.1-8b-instant",
            temperature: 0.3
        });

        const output = chat.choices[0].message.content;

        res.json({ flashcards: output });

    } catch (err) {
        console.error("Flashcards error:", err);
        res.status(500).json({ error: "Flashcards failed" });
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



app.post("/generate-cbt", async (req, res) => {
    try {
        const userId = req.headers["x-user-id"];
        if (!userQuizAttempts[userId]) {
    userQuizAttempts[userId] = {
        count: 0,
        lastReset: Date.now()
    };
        }

        // Rate limit Block
  const limit = 3;
const cooldown = 10 * 60 * 1000; // 10 minutes

const userData = userQuizAttempts[userId];

// Reset after cooldown
if (Date.now() - userData.lastReset > cooldown) {
    userData.count = 0;
    userData.lastReset = Date.now();
}

// Block if exceeded
if (userData.count >= limit) {
    const remainingTime = Math.ceil(
        (cooldown - (Date.now() - userData.lastReset)) / 60000
    );

    return res.status(429).json({
        error: `Limit reached. Try again in ${remainingTime} minute(s).`
    });
}

// Increment usage
userData.count++;
        
     //   const { filename, numQuestions = 50, difficulty } = req.body;
        let { filename, numQuestions = 20, difficulty } = req.body;

         // Enforce max limit
         numQuestions = Math.min(parseInt(numQuestions) || 20, 20);

        if (!userId || !filename) {
            return res.status(400).json({ error: "Missing data" });
        }

        // 1. GET CHUNKS FROM SUPABASE
        const { data, error } = await supabase
            .from("book_chunks")
            .select("content")
            .eq("user_id", userId)
            .eq("filename", filename);

        if (error || !data || !data.length) {
            return res.status(404).json({ error: "No content found" });
        }

        // 2. PICK RANDOM CHUNKS FOR HIGH VARIETY
        // Grab up to 40 chunks to ensure the AI has enough diverse material for 50 questions
        const chunks = data
            .sort(() => 0.5 - Math.random())
            .slice(0, Math.min(40, data.length))
            .map(c => c.content.trim());

        // Remove duplicate chunks quickly and limit string size to prevent memory/token overflows
        const uniqueChunks = [...new Set(chunks)];
        const contextText = uniqueChunks.join("\n\n").slice(0, 6000); 

        // 3. PROMPT (STRICT JSON OUTPUT, TOUGH QUESTIONS, NO REPEATS)
        const prompt = `
Act as an elite university examiner. Generate exactly ${numQuestions} multiple-choice questions based ONLY on the provided text.

DIFFICULTY LEVEL: ${difficulty ? difficulty.toUpperCase() : 'HARD'}
${difficulty === "easy" 
? "- Focus on foundational concepts, but make the distractors very plausible to test true understanding." 
: difficulty === "moderate" || difficulty === "medium"
? "- Focus on conceptual understanding and application. Questions should be tough and require critical thinking." 
: "- HARD MODE: Make the questions EXTREMELY TOUGH. Require multi-step reasoning, complex analysis, and highly subtle distractors. Test deep mastery."}

STRICT RULES:
1. GENERATE EXACTLY ${numQuestions} QUESTIONS.
2. DO NOT REPEAT QUESTIONS OR CONCEPTS. Ensure maximum variety across all ${numQuestions} questions.
3. No "What is" or simple True/False questions. 
4. Each question MUST test a different concept from the text.
5. Use ONLY the provided text.
6. Each option MUST start with A., B., C., D.
7. The answer MUST be ONLY the letter (A, B, C, or D).
8. Return ONLY valid JSON (no markdown, no extra text).

FORMAT:
{
  "questions": [
    {
      "question": "Question text",
      "options": [
        "A. Option",
        "B. Option",
        "C. Option",
        "D. Option"
      ],
      "answer": "A",
      "explanation": "Brief explanation based on the text"
    }
  ]
}

TEXT:
${contextText}
`;

const groqPrompt = `
Act as a JAMB examiner. Generate ${numQuestions} MCQs based ONLY on the provided text.

DIFFICULTY: ${difficulty.toUpperCase()}

STRICT QUESTION ARCHITECTURE:
1. NO BINARY QUESTIONS: Every question must have 4 distinct, substantive options. 
2. FORBIDDEN FORMATS: Absolutely no "True or False", "Yes or No", or "Which of these is correct".
3. MANDATORY STARTERS: Every question MUST begin with one of these specific anchors:
   - "In the context of [Concept], how does..."
   - "According to the passage, why is [Concept] described as..."
   - "Given the scenario where [X] happens, what is the impact on..."
   - "Identify the relationship between [A] and [B] regarding..."
   - "Contrast the mechanism of [A] with [B] in terms of..."
4. DISTRACTORS: All 3 wrong options must be technical terms from the text, but applied contextually incorrectly.

RULES:
- QUANTITY: Exactly ${numQuestions}.
- VARIETY: Each question must target a unique paragraph—zero concept overlap.
- NO "ALL OF THE ABOVE": Every option must be a unique, stand-alone answer.
- OUTPUT: Return ONLY raw JSON. No markdown.

FORMAT:
{"questions":[{"question":"","options":["A. ","B. ","C. ","D. "],"answer":"","explanation":""}]}

TEXT:
${contextText}`;

        console.log(`🔥 Generating ${numQuestions} ${difficulty} CBT...`);

      /*let outputText = "";

// GEMINI (PRIMARY)
try {
    console.log("⚡ Using Gemini (primary)");

    outputText = await geminiGenerate(
        prompt + "\n\nSTRICT: RETURN ONLY VALID JSON. NO MARKDOWN. NO EXTRA TEXT."
    );

    if (!outputText || outputText.length < 50) {
        throw new Error("Gemini returned empty/invalid response");
    }

} catch (geminiErr) {
    console.warn("⚠️ Gemini failed → switching to Groq");

    // GROQ (FALLBACK)
    try {
        const chat = await groq.chat.completions.create({
            messages: [{ role: "user", content: prompt }],
            model: "llama-3.1-8b-instant",
            temperature: 0.4,
            max_tokens: 3000,
            response_format: { type: "json_object" }
        });

        outputText = chat.choices?.[0]?.message?.content || "";

        if (!outputText) {
            throw new Error("Groq returned empty response");
        }

    } catch (groqErr) {
        console.error("❌ Both Gemini and Groq failed");

        return res.status(500).json({
            error: "AI generation failed. Please try again."
        });
    }
}*/

        let outputText = "";

try {
    console.log("⚡ Using Gemini (primary)");

    outputText = await Promise.race([
        geminiGenerate(prompt),
        new Promise((_, reject) =>
            setTimeout(() => reject(new Error("Gemini timeout")), 12000)
        )
    ]);

} catch (geminiErr) {

    console.warn("⚠️ Gemini failed → switching to Groq");

    try {
        const chat = await groq.chat.completions.create({
            messages: [{ role: "user", content: groqPrompt }],
            model: "llama-3.1-8b-instant",
            temperature: 0.4,
            max_tokens: 3000,
            response_format: { type: "json_object" }
        });

        outputText = chat.choices?.[0]?.message?.content || "";

    } catch (groqErr) {
        console.error("❌ Both Gemini and Groq failed");

        return res.status(500).json({
            error: "AI generation failed"
        });
    }
}

        console.log("📦 RAW AI OUTPUT GENERATED");

        let questionsJson = [];

        try {
            const cleaned = outputText
                .replace(/```json|```/g, "")
                .trim();

            const parsed = JSON.parse(cleaned);

            questionsJson = parsed.questions || [];

            questionsJson = questionsJson.map(q => {
            let answerIndex = ["A", "B", "C", "D"].indexOf(
            q.answer?.toUpperCase().trim()
             );

         return {
           ...q,
        answer: answerIndex === -1 ? 0 : answerIndex // fallback safety
         };
         });

            if (!Array.isArray(questionsJson) || questionsJson.length === 0) {
                throw new Error("Invalid questions array");
            }

        } catch (e) {
            console.error("❌ JSON Parse failed:", e);
            return res.status(500).json({ error: "AI returned bad format" });
        }

        // 6. ENSURE EXACT NUMBER (simple + safe)
        questionsJson = questionsJson.slice(0, numQuestions);

        // 7. FINAL RESPONSE
        res.json({ questions: questionsJson });

    } catch (err) {
        console.error("CBT Route Error:", err);
        res.status(500).json({ error: "CBT generation failed" });
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
