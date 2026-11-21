export interface SourceDocument {
  id: string | null;
  course: string | null;
  section: string;
  question: string;
  answer: string;
}

export interface ChatRequest {
  query: string;
  course: string | null;
  num_results?: number | null;
  enable_fallback?: boolean;
}

export interface ChatResponse {
  answer: string;
  sources: SourceDocument[];
  rewritten_query: string | null;
  used_fallback: boolean;
  course: string | null;
}

export interface CoursesResponse {
  default_course: string | null;
  courses: string[];
}

export interface HealthResponse {
  status: "ok";
  provider: string;
  default_course: string | null;
  num_documents: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceDocument[];
  rewrittenQuery?: string | null;
  usedFallback?: boolean;
  course?: string | null;
  isError?: boolean;
  timestamp: string;
}
