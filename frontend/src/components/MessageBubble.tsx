import AutoFixHighOutlinedIcon from "@mui/icons-material/AutoFixHighOutlined";
import PersonOutlineIcon from "@mui/icons-material/PersonOutline";
import SmartToyOutlinedIcon from "@mui/icons-material/SmartToyOutlined";
import { Alert, Avatar, Box, Chip, Paper, Typography } from "@mui/material";

import type { ChatMessage } from "../types/chat";
import { SourcesAccordion } from "./SourcesAccordion";

interface MessageBubbleProps {
  message: ChatMessage;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: isUser ? "row-reverse" : "row",
        alignItems: "flex-start",
        gap: 1.5,
      }}
    >
      <Avatar
        sx={{
          width: 34,
          height: 34,
          bgcolor: isUser ? "primary.main" : "secondary.main",
        }}
      >
        {isUser ? <PersonOutlineIcon fontSize="small" /> : <SmartToyOutlinedIcon fontSize="small" />}
      </Avatar>

      <Box sx={{ maxWidth: "82%", minWidth: 0 }}>
        {message.isError ? (
          <Alert severity="error" variant="outlined">
            {message.content}
          </Alert>
        ) : (
          <Paper
            elevation={1}
            sx={{
              px: 2,
              py: 1.5,
              borderRadius: 2,
              bgcolor: isUser ? "primary.main" : "background.paper",
              color: isUser ? "primary.contrastText" : "text.primary",
            }}
          >
            <Typography variant="body1" sx={{ whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
              {message.content}
            </Typography>

            {!isUser && message.usedFallback && message.rewrittenQuery && (
              <Box sx={{ mt: 1 }}>
                <Chip
                  icon={<AutoFixHighOutlinedIcon fontSize="small" />}
                  label={`Rewritten query: "${message.rewrittenQuery}"`}
                  size="small"
                  color="info"
                  variant="outlined"
                />
              </Box>
            )}

            {!isUser && message.sources && <SourcesAccordion sources={message.sources} />}
          </Paper>
        )}

        <Typography
          variant="caption"
          color="text.secondary"
          sx={{
            display: "block",
            mt: 0.5,
            textAlign: isUser ? "right" : "left",
          }}
        >
          {message.timestamp}
        </Typography>
      </Box>
    </Box>
  );
}
