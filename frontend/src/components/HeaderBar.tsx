import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import ForumOutlinedIcon from "@mui/icons-material/ForumOutlined";
import {
  AppBar,
  Box,
  Chip,
  FormControl,
  IconButton,
  InputLabel,
  MenuItem,
  Select,
  Toolbar,
  Tooltip,
  Typography,
} from "@mui/material";
import type { SelectChangeEvent } from "@mui/material/Select";

interface HeaderBarProps {
  courses: string[];
  selectedCourse: string;
  onCourseChange: (course: string) => void;
  provider: string | null;
  canClear: boolean;
  onClearChat: () => void;
}

export function HeaderBar({
  courses,
  selectedCourse,
  onCourseChange,
  provider,
  canClear,
  onClearChat,
}: HeaderBarProps) {
  const handleChange = (event: SelectChangeEvent<string>) => {
    onCourseChange(event.target.value);
  };

  return (
    <AppBar position="sticky" color="default" elevation={1}>
      <Toolbar sx={{ gap: 2, flexWrap: "wrap", py: 1 }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexGrow: 1 }}>
          <ForumOutlinedIcon color="primary" />
          <Typography variant="h6" component="h1" sx={{ fontWeight: 600 }}>
            FAQ RAG Assistant
          </Typography>
          {provider && (
            <Chip
              label={provider}
              size="small"
              variant="outlined"
              color="primary"
              sx={{ ml: 1 }}
            />
          )}
        </Box>

        <FormControl size="small" sx={{ minWidth: 200 }}>
          <InputLabel id="course-select-label">Course</InputLabel>
          <Select
            labelId="course-select-label"
            id="course-select"
            value={selectedCourse}
            label="Course"
            onChange={handleChange}
          >
            <MenuItem value="all">All courses</MenuItem>
            {courses.map((course) => (
              <MenuItem key={course} value={course}>
                {course}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <Tooltip title="Clear chat">
          <span>
            <IconButton
              aria-label="Clear chat"
              onClick={onClearChat}
              disabled={!canClear}
              size="small"
            >
              <DeleteOutlineIcon />
            </IconButton>
          </span>
        </Tooltip>
      </Toolbar>
    </AppBar>
  );
}
