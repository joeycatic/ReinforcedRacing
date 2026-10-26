// Acceptance-only static file host. It serves the exact build under a repository prefix.
import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { resolve, sep, extname } from "node:path";
const root = resolve("dist"),
  prefix = "/racing-rl/";
const types = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".wasm": "application/wasm",
  ".json": "application/json",
  ".css": "text/css",
  ".svg": "image/svg+xml",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
  ".onnx": "application/octet-stream",
};
createServer(async (req, res) => {
  const url = new URL(req.url, "http://localhost");
  if (!url.pathname.startsWith(prefix)) {
    res.writeHead(404);
    res.end();
    return;
  }
  let path = resolve(
    root,
    decodeURIComponent(url.pathname.slice(prefix.length)),
  );
  if (path !== root && !path.startsWith(root + sep)) {
    res.writeHead(404);
    res.end();
    return;
  }
  try {
    if ((await stat(path)).isDirectory()) path = resolve(path, "index.html");
    const data = await readFile(path);
    res.writeHead(200, {
      "Content-Type": types[extname(path)] ?? "application/octet-stream",
    });
    res.end(data);
  } catch {
    res.writeHead(404);
    res.end();
  }
}).listen(4174, "127.0.0.1", () =>
  console.log("Static acceptance host: http://127.0.0.1:4174/racing-rl/"),
);
