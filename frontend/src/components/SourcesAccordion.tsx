import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import LibraryBooksOutlinedIcon from "@mui/icons-material/LibraryBooksOutlined";
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Box,
  Chip,
  Divider,
  Stack,
  Typography,
} from "@mui/material";

import type { SourceDocument } from "../types/chat";

interface SourcesAccordionProps {
  sources: SourceDocument[];
}

export function SourcesAccordion({ sources }: SourcesAccordionProps) {
  if (sources.length === 0) {
    return null;
  }

  return (
    <Accordion
      disableGutters
      elevation={0}
      sx={{
        mt: 1.5,
        bgcolor: "action.hover",
        borderRadius: 1,
        "&::before": { display: "none" },
      }}
    >
      <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={{ minHeight: 38 }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <LibraryBooksOutlinedIcon fontSize="small" color="action" />
          <Typography variant="caption" sx={{ fontWeight: 600 }}>
            Sources ({sources.length})
          </Typography>
        </Box>
      </AccordionSummary>
      <AccordionDetails sx={{ pt: 0 }}>
        <Stack spacing={1.5} divider={<Divider flexItem />}>
          {sources.map((doc, idx) => (
            <Box key={doc.id ?? `${doc.course ?? "faq"}-${idx}`}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap", mb: 0.5 }}>
                {doc.course && <Chip label={doc.course} size="small" variant="outlined" />}
                {doc.section && (
                  <Typography variant="caption" color="text.secondary">
                    {doc.section}
                  </Typography>
                )}
              </Box>
              {doc.question && (
                <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.25 }}>
                  {doc.question}
                </Typography>
              )}
              {doc.answer && (
                <Typography
                  variant="caption"
                  color="text.secondary"
                  sx={{ display: "block", whiteSpace: "pre-wrap" }}
                >
                  {doc.answer}
                </Typography>
              )}
            </Box>
          ))}
        </Stack>
      </AccordionDetails>
    </Accordion>
  );
}
