import fs from "fs";
import path from "path";
import fetch from "node-fetch";
import dotenv from "dotenv";

dotenv.config({ path: ".env" });

const STARS_FILE = path.join(process.cwd(), "src", "stars.json");
const API_KEY = process.env.SERPAPI_API_KEY;

if (!API_KEY) {
  console.error("❌ Missing SERPAPI_API_KEY in .env");
  process.exit(1);
}

// Extract snippet from multiple possible fields
function extractSnippet(data) {
  const fields = [
    data.knowledge_graph?.description,
    data.knowledge_graph?.snippet,
    data.answer_box?.snippet,
    data.answer_box?.answer,
    ...(data.organic_results || []).map(r => r.snippet),
    ...(data.related_questions || []).map(r => r.snippet)
  ];

  const valid = fields.filter(Boolean);
  return valid.length > 0 ? valid.join(" ") : null;
}

// Fetch snippet for one star
async function fetchSnippet(star) {
  const query = star.search_query || star.display || star.name;
  if (!query) return null;

  const url = `https://serpapi.com/search.json?engine=google&q=${encodeURIComponent(query)}&hl=en&api_key=${API_KEY}&num=3`;

  try {
    const res = await fetch(url);
    const data = await res.json();
    const snippet = extractSnippet(data);
    if (!snippet) {
      console.warn(`⚠️ No snippet found for ${star.display}, retrying with Bing engine...`);
      // Retry with Bing if Google engine fails
      const retryUrl = `https://serpapi.com/search.json?engine=bing&q=${encodeURIComponent(query)}&api_key=${API_KEY}&num=3`;
      const retryRes = await fetch(retryUrl);
      const retryData = await retryRes.json();
      return extractSnippet(retryData) || null;
    }
    return snippet;
  } catch (err) {
    console.error(`❌ Error fetching snippet for ${star.display}:`, err.message);
    return null;
  }
}

async function run() {
  if (!fs.existsSync(STARS_FILE)) {
    console.error("❌ stars.json not found in src/");
    process.exit(1);
  }

  const raw = fs.readFileSync(STARS_FILE, "utf-8");
  const stars = JSON.parse(raw);

  for (let i = 0; i < stars.length; i++) {
    const star = stars[i];

    console.log(`🔎 Processing star (${i + 1}/${stars.length}): ${star.display}`);

    let snippet = null;
    let attempts = 0;

    // Keep retrying until we get a snippet or max 3 attempts
    while (!snippet && attempts < 3) {
      snippet = await fetchSnippet(star);
      attempts++;
    }

    if (!snippet) {
      console.error(`❌ Could not fetch snippet for ${star.display} after ${attempts} attempts.`);
      snippet = "Description unavailable. Please check online."; // very last fallback
    }

    star.snippet = snippet;
    console.log(`✅ Snippet for ${star.display}: ${snippet.slice(0, 100)}${snippet.length > 100 ? "..." : ""}`);

    // Save every 5 stars
    if (i % 5 === 0) {
      fs.writeFileSync(STARS_FILE, JSON.stringify(stars, null, 2));
    }
  }

  fs.writeFileSync(STARS_FILE, JSON.stringify(stars, null, 2));
  console.log("🎉 Done! All stars now have a snippet.");
}

run();
