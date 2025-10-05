// pages/api/starinfo.ts
//import type { NextApiRequest, NextApiResponse } from "next";

/*type StarInfo = {
  title: string;
  description: string;
  url: string | null;
};

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<StarInfo | { error: string }>
) {
  const q = (req.query.q as string || "").trim();
  if (!q) return res.status(400).json({ error: "Missing query" });

  const apiKey = process.env.SERPAPI_API_KEY;
  if (!apiKey) return res.status(500).json({ error: "Missing SERPAPI_API_KEY" });

  try {
    const url = new URL("https://serpapi.com/search.json");
    url.searchParams.set("engine", "google");
    url.searchParams.set("q", q);
    url.searchParams.set("api_key", apiKey);
    url.searchParams.set("num", "3");
    console.log("Fetching:", url.toString());

    const r = await fetch(url.toString());
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = (await r.json()) as any;
    console.log("SerpAPI response:", data);

    const kg = data.knowledge_graph || {};
    const organic = data.organic_results || [];

    // Build summary
    let description = "";
    if (kg.description) description = kg.description;
    else if (organic.length > 0)
      description = organic.map((r: any) => r.snippet).filter(Boolean).join(" ");
    if (!description) description = "No description found.";

    const payload: StarInfo = {
      title: kg.title || organic[0]?.title || q,
      description,
      url: kg.source?.url || organic[0]?.link || null,
    };

    res.setHeader("Cache-Control", "s-maxage=86400, stale-while-revalidate=86400");
    res.status(200).json(payload);
  } catch (error: unknown) {
    console.error(error);
    let message = "SerpAPI error";
    if (error instanceof Error) message = error.message;
    res.status(500).json({ error: message });
  }
}*/
