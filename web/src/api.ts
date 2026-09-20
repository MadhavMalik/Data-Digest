import type { DatasetRow } from "./types";

async function json<T>(res: Response): Promise<T> {
  const body = await res.json().catch(() => ({}) as Record<string, unknown>);
  if (!res.ok) {
    const detail = (body as { detail?: string }).detail;
    throw new Error(detail ?? `${res.status} ${res.statusText}`);
  }
  return body as T;
}

export async function listDatasets(): Promise<DatasetRow[]> {
  const data = await json<{ datasets: DatasetRow[] }>(await fetch("/datasets"));
  return data.datasets ?? [];
}

export async function uploadDataset(
  file: File,
  onProgress?: (fraction: number) => void,
): Promise<DatasetRow> {
  // XHR rather than fetch: upload progress is the one thing fetch still cannot
  // report, and a 60 MB Parquet upload with no feedback feels broken.
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("file", file);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/datasets/upload");
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      let parsed: Record<string, unknown> = {};
      try {
        parsed = JSON.parse(xhr.responseText) as Record<string, unknown>;
      } catch {
        /* fall through to the status-based error below */
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(parsed as unknown as DatasetRow);
      } else {
        reject(new Error((parsed.detail as string) ?? `upload failed (${xhr.status})`));
      }
    };
    xhr.onerror = () => reject(new Error("network error during upload"));
    xhr.send(form);
  });
}

export interface StartRequest {
  question: string;
  path: string | null;
  max_rounds: number;
  max_visualizations: number;
}

export async function startAnalysis(req: StartRequest): Promise<string> {
  const data = await json<{ analysis_id: string }>(
    await fetch("/analyses", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...req, enable_web_grounding: false }),
    }),
  );
  return data.analysis_id;
}

export interface Health {
  config: {
    llm: { model: string; configured: boolean };
    elastic: { configured: boolean };
    brave: { configured: boolean };
  };
  missing_credentials: string[];
}

export async function getHealth(): Promise<Health> {
  return json<Health>(await fetch("/health"));
}
