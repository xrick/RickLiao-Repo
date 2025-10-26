## Project Structure Template of 12-Factor Principles 
/ai-agent/
│
├── config/
│   ├── env.prod.yaml
│   ├── env.dev.yaml
│   └── prompt_config.yaml
│
├── core/
│   ├── agent.py          # Stateless decision logic
│   ├── memory.py         # Vector-based memory
│   └── planner.py        # Task orchestration
│
├── skills/
│   ├── retrieval.py
│   ├── summarizer.py
│   └── product_matcher.py
│
├── tests/
│   └── prompt_tests/
│       ├── intent_parsing_test.json
│       └── entity_extraction_test.json
│
└── telemetry/
    ├── trace_logger.py
    └── rlaif_feedback.py
