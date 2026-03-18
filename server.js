require("dotenv").config();

const express = require("express");
const cors = require("cors");
const fs = require("fs");
const path = require("path");
const pdfParse = require("pdf-parse");
const Groq = require("groq-sdk");
const natural = require("natural");
const { createClient } = require("@supabase/supabase-js");
const { S3Client, PutObjectCommand, GetObjectCommand } = require("@aws-sdk/client-s3");
const { getSignedUrl } = require("@aws-sdk/s3-request-presigner");
const { pipeline } = require("@xenova/transformers");
const mammoth = require("mammoth");
const EventEmitter = require("events");

const app = express();
const progressEvents = new EventEmitter();

app.use(cors());
app.use(express.json({ limit: "10mb" }));

// --- INIT ---
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY);

const r2 = new S3Client({
  region: "auto",
  endpoint: `https://${process.env.R2_ACCOUNT_ID}.r2.cloudflarestorage.com`,
  credentials: {
    accessKeyId: process.env.R2_ACCESS_KEY_ID,
    secretAccessKey: process.env.R2_SECRET_ACCESS_KEY,
  },
});

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

let embedder;

// --- LOAD MODEL ---
(async () => {
  console.log("Loading embedding model...");
  embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", {
    quantized: true,
  });
  console.log("✅ Model ready");
})();

// --- HELPERS ---

async function embedText(text) {
  const safe = text.slice(0, 2000); // ✅ GOOD EMBEDDING
  const result = await embedder(safe, { pooling: "mean", normalize: true });
  return Array.from(result.data);
}

function chunkText(text, size = 700, overlap = 100) {
  const words = text.split(/\s+/);
  let chunks = [];

  for (let i = 0; i < words.length; i += size - overlap) {
    chunks.push(words.slice(i, i + size).join(" "));
  }

  return chunks.slice(0, 200); // ✅ BALANCED LIMIT
}

function buildContext(chunks, maxChars = 1800) {
  let context = "";
  let used = 0;

  for (const c of chunks) {
    if (!c) continue;
    if (used + c.length > maxChars) break;
    context += c + "\n\n---\n\n";
    used += c.length;
  }

  return context.trim();
}

function analyzeQuery(q) {
  q = q.toLowerCase();
  return {
    isSummary: /summary|overview/.test(q),
    isShort: /brief|short/.test(q),
    isDefinition: /define|what is/.test(q),
  };
}

function reciprocalRankFusion(v, k) {
  const scores = new Map();

  function add(list) {
    list.forEach((item, i) => {
      const key = item.text;
      const score = 1 / (60 + i);
      if (!scores.has(key)) scores.set(key, { text: key, score: 0 });
      scores.get(key).score += score;
    });
  }

  add(v);
  add(k);

  return [...scores.values()]
    .sort((a, b) => b.score - a.score)
    .map(x => x.text);
}

// --- GROQ RETRY ---
async function callGroqWithRetry(payload, retries = 3) {
  try {
    return await groq.chat.completions.create(payload);
  } catch (err) {
    if (err.status === 429 && retries > 0) {
      console.log("⏳ Rate limited... retrying");
      await new Promise(r => setTimeout(r, 2000));
      return callGroqWithRetry(payload, retries - 1);
    }
    throw err;
  }
}

// --- UPLOAD ---

app.post("/generate-upload-url", async (req, res) => {
  const userId = req.headers["x-user-id"];
  const { filename } = req.body;

  const key = `${userId}/${Date.now()}_${filename}`;
  const command = new PutObjectCommand({ Bucket: process.env.R2_BUCKET, Key: key });
  const uploadUrl = await getSignedUrl(r2, command, { expiresIn: 600 });

  res.json({ uploadUrl, fileKey: key });
});

app.post("/upload", async (req, res) => {
  const { fileKey, filename } = req.body;
  const userId = req.headers["x-user-id"];
  const temp = path.join(__dirname, "tmp-" + Date.now());

  try {
    const file = await r2.send(new GetObjectCommand({
      Bucket: process.env.R2_BUCKET,
      Key: fileKey,
    }));

    await new Promise((resolve, reject) => {
      const ws = fs.createWriteStream(temp);
      file.Body.pipe(ws);
      ws.on("finish", resolve);
      ws.on("error", reject);
    });

    progressEvents.emit(userId, { step: "Extracting...", progress: 30 });

    // ✅ MEMORY SAFETY LIMIT
    const stats = fs.statSync(temp);
    const MAX_SIZE_MB = 25; // safer than 50

    if (stats.size > MAX_SIZE_MB * 1024 * 1024) {
      throw new Error("File too large. Max 25MB allowed.");
    }

    let text = "";

    if (filename.endsWith(".pdf")) {
      const buffer = fs.readFileSync(temp);

      const data = await pdfParse(buffer, {
        max: 120
      });

      text = data.text;
    }

    if (filename.endsWith(".docx")) {
      const data = await mammoth.extractRawText({ path: temp });
      text = data.value;
    }

    if (!text || text.length < 50) throw new Error("Bad file");

    const chunks = chunkText(text);

    progressEvents.emit(userId, { step: "Embedding...", progress: 50 });

    const batchSize = 20;

    for (let i = 0; i < chunks.length; i += batchSize) {
      const batch = chunks.slice(i, i + batchSize);

      const vectors = await Promise.all(batch.map(embedText));

      const rows = batch.map((c, idx) => ({
        user_id: userId,
        filename,
        content: c,
        embedding: vectors[idx],
      }));

      await supabase.from("book_chunks").insert(rows);

      const prog = 50 + Math.floor((i / chunks.length) * 40);
      progressEvents.emit(userId, { step: "Saving...", progress: prog });
    }

    await supabase.from("books").insert([{ user_id: userId, filename }]);

    progressEvents.emit(userId, { step: "Completed", progress: 100 });

    res.json({ success: true });

  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e.message || "Upload failed" });
  } finally {
    if (fs.existsSync(temp)) fs.unlinkSync(temp);
  }
});

// --- CONTEXT ---

async function getContext(userId, topic, book) {
  const vec = await embedText(topic);

  const { data } = await supabase.rpc("match_book_chunks", {
    query_embedding: vec,
    match_threshold: 0,
    match_count: 5,
    p_user_id: userId,
    p_filename: book === "all" ? null : book,
  });

  if (!data || data.length === 0) return [];

  const vector = data.map(d => d.content);

  const tfidf = new natural.TfIdf();
  vector.forEach(c => tfidf.addDocument(c));

  let keyword = [];
  tfidf.tfidfs(topic, (i, m) => {
    keyword.push({ text: vector[i], score: m });
  });

  keyword.sort((a, b) => b.score - a.score);
  keyword = keyword.slice(0, 2);

  const combined = reciprocalRankFusion(
    vector.map(t => ({ text: t })),
    keyword
  );

  return combined.slice(0, 5);
}

// --- AI CORE ---

async function runAI(type, req, res) {
  const { topic, book = "all" } = req.body;
  const userId = req.headers["x-user-id"];
  const queryType = analyzeQuery(topic);

  // ✅ CACHE FIXED
  const { data: cached } = await supabase
    .from("ai_cache")
    .select("response")
    .eq("user_id", userId)
    .eq("topic", topic)
    .eq("type", type)
    .eq("book", book)
    .maybeSingle();

  if (cached) return res.json({ [type]: cached.response });

  const chunks = await getContext(userId, topic, book);

  // ✅ EMPTY CHECK FIXED
  if (!chunks || chunks.length === 0) {
    return res.json({ [type]: "No relevant content found." });
  }

  let maxChars = 1800;
  if (queryType.isSummary) maxChars = 2500;
  if (queryType.isShort) maxChars = 1000;

  const context = buildContext(chunks, maxChars);

  let prompt;

  if (type === "explanation") {
    prompt = `
Explain clearly using the context.

CONTEXT:
${context}

TOPIC:
${topic}

- Use headings
- Be detailed
`;
  }

  else if (type === "notes") {
    prompt = `
Create structured study notes.

CONTEXT:
${context}

TOPIC:
${topic}

FORMAT:

# Overview
# Key Concepts
# Processes
# Summary Points
# Exam Tips
`;
  }

  else if (type === "quiz") {
    prompt = `
Create a quiz.

CONTEXT:
${context}

TOPIC:
${topic}

- 10 questions
- 4 options
- correct answer
- explanation
`;
  }

  const chat = await callGroqWithRetry({
    model: "llama-3.1-8b-instant",
    messages: [{ role: "user", content: prompt }],
    temperature: 0.2,
  });

  const output = chat.choices[0].message.content;

  // ✅ CACHE SAVE FIXED
  await supabase.from("ai_cache").insert({
    user_id: userId,
    topic,
    type,
    book,
    response: output,
  });

  res.json({ [type]: output });
}

// --- ROUTES ---
app.post("/deep-explain", (req, res) => runAI("explanation", req, res));
app.post("/notes", (req, res) => runAI("notes", req, res));
app.post("/quiz", (req, res) => runAI("quiz", req, res));

// --- PROGRESS ---
app.get("/progress", (req, res) => {
  const userId = req.query.userId;

  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Connection": "keep-alive",
  });

  const handler = data => {
    res.write(`data: ${JSON.stringify(data)}\n\n`);
  };

  progressEvents.on(userId, handler);

  req.on("close", () => {
    progressEvents.removeListener(userId, handler);
  });
});

app.get("/books", async (req, res) => {
  const userId = req.headers["x-user-id"];

  const { data, error } = await supabase
    .from("books")
    .select("*")
    .eq("user_id", userId);

  if (error) return res.status(500).json([]);

  res.json(data || []);
});

// --- SERVER ---
app.listen(3000, () => console.log("🚀 Running"));
