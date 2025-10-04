// scripts/fetchStarSnippets.js
import fs from "fs";
import path from "path";
import fetch from "node-fetch";
import dotenv from "dotenv";

dotenv.config();

const API_KEY = process.env.SERPAPI_API_KEY;
if (!API_KEY) throw new Error("Missing SERPAPI_API_KEY in .env");

const starsPath = path.resolve("./src/stars.json");

async function fetchSnippet(star) {
  try {
    const url = new URL("https://serpapi.com/search.json");
    url.searchParams.set("engine", "google");
    url.searchParams.set("q", star.search_query);
    url.searchParams.set("api_key", API_KEY);
    url.searchParams.set("num", "3"); // fetch a few results

    const res = await fetch(url.toString());
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    const kg = data.knowledge_graph || {};
    const organic = data.organic_results || [];

    let description = "";
    if (kg.description) description = kg.description;
    else if (organic.length > 0)
      description = organic.map((r) => r.snippet).filter(Boolean).join(" ");

    if (!description) description = "No description available.";

    return description;
  } catch (err) {
    console.error(`Error fetching ${star.id}:`, err.message);
    return "No description available.";
  }
}

async function main() {
  const raw = fs.readFileSync(starsPath, "utf-8");
  const stars = JSON.parse(raw);

  for (let i = 0; i < stars.length; i++) {
    const star = stars[i];
    console.log(`Fetching snippet for ${star.display} (${i + 1}/${stars.length})...`);
    const snippet = await fetchSnippet(star);
    stars[i].snippet = snippet;
  }

  fs.writeFileSync(starsPath, JSON.stringify(stars, null, 2));
  console.log("All snippets added to stars.json!");
}

main();
