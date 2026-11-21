import SendIcon from "@mui/icons-material/Send";
import { Box, CircularProgress, IconButton, Paper, TextField } from "@mui/material";
import type { FormEvent, KeyboardEvent } from "react";
import { useState } from "react";

interface ChatInputProps {
  disabled: boolean;
  onSend: (query: string) => void;
}

const MAX_QUERY_CHARS = 2000;

export function ChatInput({ disabled, onSend }: ChatInputProps) {
  const [value, setValue] = useState("");

  const submit = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled) {
      return;
    }
    onSend(trimmed);
    setValue("");
  };

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    submit();
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  return (
    <Paper
      component="form"
      onSubmit={handleSubmit}
      elevation={3}
      sx={{
        p: 1.5,
        borderRadius: 2,
        display: "flex",
        alignItems: "flex-end",
        gap: 1,
      }}
    >
      <TextField
        fullWidth
        multiline
        maxRows={4}
        placeholder="Ask a question about DataTalks.Club courses..."
        value={value}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        slotProps={{ htmlInput: { maxLength: MAX_QUERY_CHARS } }}
        size="small"
      />
      <Box sx={{ display: "flex", alignItems: "center" }}>
        <IconButton
          type="submit"
          color="primary"
          aria-label="Send message"
          disabled={disabled || value.trim().length === 0}
        >
          {disabled ? <CircularProgress size={22} /> : <SendIcon />}
        </IconButton>
      </Box>
    </Paper>
  );
}
