import {
  Alert,
  Box,
  Chip,
  Container,
  Paper,
  Stack,
  Typography,
} from "@mui/material";
import { useEffect, useRef, useState } from "react";

import { fetchCourses, fetchHealth, sendChatMessage } from "./api/client";
import { ChatInput } from "./components/ChatInput";
import { HeaderBar } from "./components/HeaderBar";
import { MessageBubble } from "./components/MessageBubble";
import type { ChatMessage } from "./types/chat";

const STARTER_QUESTIONS = [
  "Can I join the course after the start date?",
  "How do I run Ollama locally with Docker?",
  "What is hybrid search and RRF?",
];

function formatTime(): string {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [courses, setCourses] = useState<string[]>([]);
  const [selectedCourse, setSelectedCourse] = useState<string>("llm-zoomcamp");
  const [provider, setProvider] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [initError, setInitError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let active = true;
    Promise.all([fetchHealth(), fetchCourses()])
      .then(([health, courseData]) => {
        if (!active) {
          return;
        }
        setProvider(health.provider);
        setCourses(courseData.courses);
        setSelectedCourse(courseData.default_course ?? "all");
        setInitError(null);
      })
      .catch((error: unknown) => {
        if (!active) {
          return;
        }
        const message =
          error instanceof Error ? error.message : "Failed to connect to the FAQ API.";
        setInitError(message);
      });

    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const handleSend = async (query: string) => {
    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: query,
      timestamp: formatTime(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);

    try {
      const response = await sendChatMessage({
        query,
        course: selectedCourse === "all" ? null : selectedCourse,
      });
      const assistantMessage: ChatMessage = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: response.answer,
        sources: response.sources,
        rewrittenQuery: response.rewritten_query,
        usedFallback: response.used_fallback,
        course: response.course,
        timestamp: formatTime(),
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error: unknown) {
      const errorText = error instanceof Error ? error.message : "Unexpected request error.";
      const errorMessage: ChatMessage = {
        id: `error-${Date.now()}`,
        role: "assistant",
        content: errorText,
        isError: true,
        timestamp: formatTime(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
      sx={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        bgcolor: "grey.50",
      }}
    >
      <HeaderBar
        courses={courses}
        selectedCourse={selectedCourse}
        onCourseChange={setSelectedCourse}
        provider={provider}
        canClear={messages.length > 0 && !loading}
        onClearChat={() => setMessages([])}
      />

      <Container
        maxWidth="md"
        sx={{
          flexGrow: 1,
          display: "flex",
          flexDirection: "column",
          py: 2.5,
          gap: 2,
        }}
      >
        {initError && (
          <Alert severity="warning">
            Unable to reach backend API: {initError}. Start it with{" "}
            <code>uv run llm-rag-faq-api --reload</code>.
          </Alert>
        )}

        <Box sx={{ flexGrow: 1, overflowY: "auto", pr: 0.5 }}>
          {messages.length === 0 ? (
            <Paper
              variant="outlined"
              sx={{
                p: 4,
                mt: 4,
                textAlign: "center",
                borderRadius: 2,
                bgcolor: "background.paper",
              }}
            >
              <Typography variant="h6" gutterBottom>
                Ask anything about DataTalks.Club courses
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2.5 }}>
                Answers are grounded in course FAQ documents using hybrid lexical + semantic search.
              </Typography>
              <Stack
                direction="row"
                spacing={1}
                useFlexGap
                flexWrap="wrap"
                justifyContent="center"
              >
                {STARTER_QUESTIONS.map((question) => (
                  <Chip
                    key={question}
                    label={question}
                    onClick={() => void handleSend(question)}
                    disabled={loading}
                    clickable
                    variant="outlined"
                  />
                ))}
              </Stack>
            </Paper>
          ) : (
            <Stack spacing={2.5}>
              {messages.map((message) => (
                <MessageBubble key={message.id} message={message} />
              ))}
              <div ref={bottomRef} />
            </Stack>
          )}
        </Box>

        <ChatInput disabled={loading} onSend={(q) => void handleSend(q)} />
      </Container>
    </Box>
  );
}

export default App;
