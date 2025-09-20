DROP TABLE IF EXISTS evaluations;

CREATE TABLE evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    resume_filename TEXT NOT NULL,
    jd_filename TEXT NOT NULL,
    relevance_score INTEGER NOT NULL,
    verdict TEXT NOT NULL,
    feedback TEXT,
    matched_skills TEXT,
    missing_skills TEXT,
    analysis_type TEXT
);
