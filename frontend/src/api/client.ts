import type {
  ChatRequest,
  ChatResponse,
  CoursesResponse,
  HealthResponse,
} from "../types/chat";

async function parseError(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as {
      detail?: string | Array<{ msg?: string }>;
    };
    if (typeof body.detail === "string") {
      return body.detail;
    }
    if (Array.isArray(body.detail) && body.detail.length > 0) {
      return body.detail.map((item) => item.msg ?? "Validation error").join("; ");
    }
  } catch {
    // Fall back to HTTP status text when response is not JSON.
  }
  return `Request failed (${response.status} ${response.statusText})`;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const response = await fetch("/api/health");
  if (!response.ok) {
    throw new Error(await parseError(response));
  }
  return (await response.json()) as HealthResponse;
}

export async function fetchCourses(): Promise<CoursesResponse> {
  const response = await fetch("/api/courses");
  if (!response.ok) {
    throw new Error(await parseError(response));
  }
  return (await response.json()) as CoursesResponse;
}

export async function sendChatMessage(payload: ChatRequest): Promise<ChatResponse> {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(await parseError(response));
  }
  return (await response.json()) as ChatResponse;
}
