# Plan Management

Use this directory to track implementation plans for work inside `Emanuele/`.

## Directory Structure

- `current plans/` contains active plans that still have pending tasks.
- `old plans/` contains completed, superseded, cancelled, or no-longer-relevant plans.

## Creating a Plan

Create each plan as a Markdown file under `current plans/`. Use a descriptive lowercase filename, for example:

```text
current plans/review-a-action-plan.md
current plans/reproducibility-cleanup.md
```

Each plan should identify:

1. The source issue, review, or objective.
2. Concrete actions and affected files.
3. Feasibility, difficulty, and estimated effort.
4. Dependencies or decisions that may block progress.
5. Completion criteria.

Update task status in the plan as work progresses. Keep estimates explicit when scope changes.

## Archiving a Plan

Move a plan to `old plans/` when all required work is complete or when the plan is replaced or abandoned. Add a short status note near the top stating:

- final status;
- completion or archive date;
- replacement plan, if applicable;
- known unfinished items.

Do not delete old plans unless they contain incorrect or sensitive information. They provide decision history and explain why repository changes were made.
