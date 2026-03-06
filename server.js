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
const { pipeline } = require("@xenova/transformers");
const tokenizer = new natural.WordTokenizer();

const app = express();

app.use(cors({
  origin: "*",
  methods: ["GET","POST","DELETE"],
  allowedHeaders: ["Content-Type"]
}));
app.get("/", (req, res) => {
    res.send("Study AI backend is running.");
});
app.use(express.json({ limit: "500mb" }));
app.use(express.urlencoded({ extended: true, limit: "500mb" }));

app.use((req, res, next) => {
    req.setTimeout(600000);
    next();
});

const groq = new Groq({
    apiKey: process.env.GROQ_API_KEY
});

const UPLOAD_DIR = "./uploads";
const CACHE_FILE = "./rag_cache.json";

if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR);

/* ---------------------- */
/* MULTER CONFIG */
/* ---------------------- */

const storage = multer.diskStorage({
    destination: (req, file, cb) => cb(null, UPLOAD_DIR),
    filename: (req, file, cb) => {
        const safe = file.originalname.replace(/\s+/g, "_");
        cb(null, Date.now() + "_" + safe);
    }
});

const upload = multer({
    storage,
    limits: { fileSize: 500 * 1024 * 1024 },
    fileFilter: (req, file, cb) => {
        const allowed = [
            "application/pdf",
            "text/plain",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "image/jpeg",
            "image/png",
            "image/webp"
        ];

        if (allowed.includes(file.mimetype)) cb(null, true);
        else cb(new Error("Unsupported file type"), false);
    }
});

/* ---------------------- */
/* MEMORY */
/* ---------------------- */

let tfidf = new natural.TfIdf();
let paragraphs = [];
let documentStore = {};
let invertedIndex = {};

let paragraphEmbeddings = [];
let embedder;
let embeddings = [];

/* ---------------------- */
/* CACHE SYSTEM */
/* ---------------------- */

function saveCache() {
    fs.writeFileSync(CACHE_FILE, JSON.stringify(documentStore));
}

function loadCache() {
    if (fs.existsSync(CACHE_FILE)) {
        try {
            documentStore = JSON.parse(fs.readFileSync(CACHE_FILE, "utf8"));
            rebuildIndex();
        } catch {
            documentStore = {};
        }
    }
}

/* ---------------------- */
/* REBUILD INDEX */
/* ---------------------- */

async function rebuildIndex() {

  tfidf = new natural.TfIdf();
  paragraphs = [];
  paragraphEmbeddings = [];
  invertedIndex = {};

  for (const [file, paras] of Object.entries(documentStore)) {

    for (const p of paras) {

      const text = p.toLowerCase();
      const index = paragraphs.length;

      paragraphs.push({
        text,
        source: file
      });

      // TF-IDF indexing
      tfidf.addDocument(text);

      // Vector embedding
      const vector = await embedText(p);
      paragraphEmbeddings.push(vector);

      // Token index (for keyword search)
      const tokens = tokenizer.tokenize(text);

      tokens.forEach(t => {
        if (!invertedIndex[t]) invertedIndex[t] = [];
        invertedIndex[t].push(index);
      });

    }

  }

  console.log(`📚 Index rebuilt: ${paragraphs.length} chunks`);

}

/* ---------------------- */
/* ROBUST TEXT EXTRACTION */
/* ---------------------- */

async function extractText(file) {

    const ext = path.extname(file.path).toLowerCase();

    try {

        /* ---------- PDF PARSER FIX ---------- */

        if (ext === ".pdf") {

            const buffer = new Uint8Array(
                fs.readFileSync(file.path)
            );

            let result = "";

            try {

                if (typeof pdfParse === "function") {

                    const data = await pdfParse(buffer);
                    result = data?.text || "";

                }

                else if (pdfParse.PDFParse) {

                    const parser = new pdfParse.PDFParse(buffer);
                    const data = await parser.getText();

                    if (typeof data === "string") result = data;
                    else if (data?.text) result = data.text;

                }

                else if (pdfParse.default) {

                    const data = await pdfParse.default(buffer);
                    result = data?.text || "";

                }

            } catch {
                console.log("pdf-parse failed, trying textract...");
            }

            if (result && result.trim().length > 100)
                return result;

        }

        /* ---------- TEXTRACT FALLBACK ---------- */

        return new Promise(resolve => {

            textract.fromFileWithPath(
                file.path,
                { preserveLineBreaks: true },
                (err, text) => {

                    if (err) return resolve("");

                    resolve(text || "");

                }
            );

        });

    } catch {

        return "";

    }
}

/* -------------------------- */
/* QUERY EXPANSION */
/* -------------------------- */

function expandQuery(query) {

  const expansions = [
    query,
    query + " explanation",
    query + " definition",
    query + " stages",
    query + " process"
  ];

  return expansions;

}

/* -------------------------- */
/* SEMANTIC SCORE */
/* -------------------------- */

function semanticScore(query, text) {

    const qWords = query.toLowerCase().split(/\W+/);
    const tWords = text.toLowerCase().split(/\W+/);

    let overlap = 0;

    qWords.forEach(w => {
      if (tWords.includes(w)) overlap++;
  });

    return overlap / qWords.length;

}

/* ---------------------- */
/* SEARCH */
/* ---------------------- */

function searchContext(query, selectedBook = "all") {

    let scores = [];

    const tokens = tokenizer.tokenize(query.toLowerCase());
    let candidateSet = new Set();

    tokens.forEach(t => {

      if (invertedIndex[t]) {
        invertedIndex[t].forEach(i => candidateSet.add(i));
      }

    });

    const queries = expandQuery(query.toLowerCase());

    queries.forEach(q => {

     candidateSet.forEach(i => {

    const measure = tfidf.tfidf(q, i);

    if (
      paragraphs[i] &&
      (selectedBook === "all" || paragraphs[i].source === selectedBook)
    ) {

      const semantic = semanticScore(q, paragraphs[i].text);

      scores.push({
        text: paragraphs[i].text,
        score: measure + semantic
      });

    }

  });

});

    scores = scores.filter(
      (v, i, a) => a.findIndex(t => t.text === v.text) === i
    );

    const best = scores
        .sort((a, b) => b.score - a.score)
        .slice(0, 8);

    const isFound = best.length > 0 && best[0].score > 0.01;

    return {
        context: best
            .map(r => r.text)
            .join("\n\n")
            .slice(0, 10000),
        isFound
    };
}

/* ---------------------- */
/* AI CALL */
/* ---------------------- */

async function askAI(prompt, system) {

    try {

        const chat = await groq.chat.completions.create({

            model: "llama-3.1-8b-instant",

            messages: [
                { role: "system", content: system },
                { role: "user", content: prompt }
            ]

        });

        return chat.choices[0].message.content;

    } catch {

        return "⚠️ AI request failed.";

    }
}

/* ---------------------- */
/* PROMPT */
/* ---------------------- */

const PROFESSOR_SYSTEM_PROMPT = `
You are an expert university professor helping a student study from textbooks.

Use ONLY the provided textbook context.

Rules:
- Start directly with the explanation.
- Provide very detailed explanations.
- Expand concepts thoroughly.
- Break explanations into logical sections.
- Explain definitions, mechanisms, and processes.
- Use bullet points where helpful.
- Use clear academic language.
- Do NOT invent information outside the textbook.
- If the textbook context is insufficient say:
"The textbook does not provide enough information."
`;

/* -------------------------- */
/* EMBEDDING MODEL */
/* -------------------------- */

async function loadEmbedder() {

  console.log("Loading embedding model...");

  embedder = await pipeline(
    "feature-extraction",
    "Xenova/all-MiniLM-L6-v2"
  );

  console.log("Embedding model ready");

}

async function embedText(text) {

  const result = await embedder(text, {
    pooling: "mean",
    normalize: true
  });

  return Array.from(result.data);

}

function cosineSimilarity(a, b) {

  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  return dot / (Math.sqrt(normA) * Math.sqrt(normB));

}

async function semanticSearch(query) {

  const queryVector = await embedText(query);

  const scores = paragraphEmbeddings.map((vec, index) => ({
    index,
    score: cosineSimilarity(queryVector, vec)
  }));

  scores.sort((a, b) => b.score - a.score);

  return scores.slice(0, 5).map(s => paragraphs[s.index].text);

}

/* ---------------------- */
/* AI ROUTES */
/* ---------------------- */

app.post("/explain", async (req, res) => {

    const { topic, book } = req.body;

    const contextChunks = await semanticSearch(topic);
    const context = contextChunks.join("\n\n")

    const prompt = `
Context from library:

${context}

Explain "${topic}" in very detailed academic depth.

Cover:
- definition
- mechanisms
- important steps
- key concepts
- examples if present in the text

Use structured sections and bullet points.
`;

    const answer = await askAI(prompt, PROFESSOR_SYSTEM_PROMPT);

    res.json({ explanation: answer });
});

app.post("/notes", async (req, res) => {

    const { topic, book } = req.body;

    const search = searchContext(topic, book);

    const prompt = `
Context from library:

${search.context}

Create structured study notes for "${topic}".
`;

    const answer = await askAI(prompt, PROFESSOR_SYSTEM_PROMPT);

    res.json({ notes: answer });
});

app.post("/quiz", async (req, res) => {

    const { topic, book } = req.body;

    const search = searchContext(topic, book);

    const prompt = `
Context from library:

${search.context}

Create a difficult 5-question MCQ quiz about "${topic}".
`;

    const answer = await askAI(prompt, PROFESSOR_SYSTEM_PROMPT);

    res.json({ quiz: answer });
});

/* -------------------------- */
/* HEALTH CHECK */
/* -------------------------- */

app.get("/health", (req, res) => {
  res.json({ status: "alive" });
});

/* ---------------------- */
/* BOOK ROUTES */
/* ---------------------- */

app.get("/books", (req, res) => {

    const books = Object.keys(documentStore).map(name => ({

        name,

        size: fs.existsSync(path.join(UPLOAD_DIR, name))
            ? fs.statSync(path.join(UPLOAD_DIR, name)).size
            : 0

    }));

    res.json(books);
});


/* -------------------------- */
/* INDEX HELPERS */
/* -------------------------- */

function addToIndex(name, chunks) {
  chunks.forEach(chunk => {
    const tokens = tokenizer.tokenize(chunk);

    tokens.forEach(token => {

      if (!invertedIndex[token]) {
        invertedIndex[token] = [];
      }

      invertedIndex[token].push(index.length);

    });

    const tf = {};
    tokens.forEach(t => {
      tf[t] = (tf[t] || 0) + 1;
    });

    index.push({
      doc: name,
      text: chunk,
      tf
    });
  });
}

function removeFromIndex(name) {
  index = index.filter(entry => entry.doc !== name);
}

/* ---------------------- */
/* UPLOAD */
/* ---------------------- */

app.post("/upload", upload.single("book"), async (req, res) => {

    try {
        console.log("FILE RECEIVED:", req.file);

        if (!req.file)
            return res.status(400).json({ error: "No file uploaded." });

        let text = await extractText(req.file);

        // Prevent memory overload
        if (text.length > 2000000) {
        text = text.slice(0, 2000000);
        }

        if (!text || text.length < 100) {

            fs.unlinkSync(req.file.path);

            return res.status(422).json({
                error: "Could not extract text. File may be scanned image."
            });

        }

        const chunks = text
            .split(/\n\s*\n/)
            .map(p => p.trim())
            .filter(p => p.length > 40)
            .slice(0, 2000);

        documentStore[req.file.filename] = chunks;

        saveCache();
        // rebuild only if needed
        setImmediate(() => {
        try {
        addToIndex(req.file.filename, chunks);
    }   catch (err) {
        console.error("Index rebuild failed:", err);
    }

});
        res.json({
            name: req.file.filename,
            chunks: chunks.length
        });

    }  catch (err) {
       console.error(err);
       res.status(500).json({ error: "Upload failed." });
    }
});

/* ---------------------- */
/* DELETE BOOK */
/* ---------------------- */

app.delete("/delete-book/:name", (req, res) => {

    const name = decodeURIComponent(req.params.name);

    if (documentStore[name]) {

        delete documentStore[name];

        const filePath = path.join(UPLOAD_DIR, name);

        if (fs.existsSync(filePath))
            fs.unlinkSync(filePath);

        saveCache();
        removeFromIndex(name);

        return res.json({ success: true });

    }

    res.status(404).json({ error: "Not found" });
});

/* ---------------------- */
/* START SERVER */
/* ---------------------- */

app.get("/health", (req, res) => {
  res.json({ status: "alive" });
});

const PORT = process.env.PORT || 5000;

app.listen(PORT, async () => {

    await loadEmbedder();
   
    loadCache();
 
    console.log("🚀 Server running on port " + PORT);

});
