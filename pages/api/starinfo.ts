// pages/api/starinfo.ts
import type { NextApiRequest, NextApiResponse } from "next";

type InfoResponse = {
  title: string;
  description: string;
  url: string | null;
};

type ErrorResponse = {
  error: string;
};

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<InfoResponse | ErrorResponse>
) {
  const q = (req.query.q as string || "").trim();
  if (!q) {
    return res.status(400).json({ error: "Missing query parameter 'q'" });
  }

  const apiKey = process.env.SERPAPI_API_KEY;
  if (!apiKey) {
    return res.status(500).json({ error: "Missing SERPAPI_API_KEY in .env.local" });
  }

  try {
    const url = new URL("https://serpapi.com/search.json");
    url.searchParams.set("engine", "google");
    url.searchParams.set("q", q);
    url.searchParams.set("api_key", apiKey);
    url.searchParams.set("num", "1"); // just the first result

    const r = await fetch(url.toString());
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();

    // Prefer knowledge_graph, then first organic result
    const kg = data.knowledge_graph ?? {};
    const organic = Array.isArray(data.organic_results) ? data.organic_results : [];
    const firstOrganic = organic[0];

    const payload: InfoResponse = {
      title: kg.title || firstOrganic?.title || q,
      description: kg.description || firstOrganic?.snippet || "No description found.",
      url: kg.source?.url || firstOrganic?.link || null,
    };

    // Cache at CDN for 1 day
    res.setHeader("Cache-Control", "s-maxage=86400, stale-while-revalidate=86400");
    res.status(200).json(payload);
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : "Unknown error";
    res.status(500).json({ error: msg });
  }
}
