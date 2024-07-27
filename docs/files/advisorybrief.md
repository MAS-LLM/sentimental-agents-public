# Templates and Data models

## Overview

The `advisory_brief.py` file contains templates and data models for creating advisory briefs in the context of simulating dialogues involving various advisors. The briefs include topics, advisor priorities, descriptions, evaluation criteria, profiles, and system messages. These templates guide the behavior and communication style of the advisors during the simulation.

## Constants and Templates

### `TOPIC`

Defines a template for the main topic of the conversation, involving the pros and cons of hiring a candidate for a specific role. It includes placeholders for candidate information, role details, and a description section.

### `ADVISOR_PRIORITIES`

Defines a template for advisors to reply with their main objectives and priorities. It includes placeholders for the advisor's name, word limit, and encourages direct and concise communication.

### `ADVISOR_DESCRIPTION`

Defines a template for advisors to reply with a description of themselves. It encourages a balanced and professional tone while providing information about the advisor's experience and expertise.

### `ADVISOR_CRITERIA`

Defines a template for advisors to reply with a list of evaluation criteria. It emphasizes the relevance to the advisor's function, objectives, and priorities, while encouraging originality and conciseness.

### `ADVISOR_PROFILE`

Defines a template for advisors to reply with a complete profile. It includes placeholders for the advisor's name, short bio, objectives and priorities, and evaluation criteria.

### `SYSTEM_MESSAGE` and `SYSTEM_MESSAGE2`

Define templates for system messages that guide the behavior of advisors during the conversation. It includes instructions for providing perspectives, citing sources, and maintaining specific roles.

### `SPECIFIC_TOPIC`

Defines a template for the facilitator to make the conversation topic more specific. It encourages a concise response directly addressing the participants.

### `ANALYTICS_TEMPLATE`

Defines a template for generating an analytics report. It instructs a psychologist to analyze discussion data, identifying firm convictions, agreeable opinions, and subjects with significant shifts in sentiment or perspective.

## Usage

1. **Importing**: Import the necessary constants and templates from `advisory_brief.py` into other Python files.

2. **Template Substitution**: Substitute placeholders in the templates with actual data when generating messages for advisors during the simulation.

3. **Customization**: Modify or extend the templates based on specific requirements or to adapt to different simulation scenarios.

4. **Note**: Ensure that the templates are used consistently and appropriately in the simulation process, guiding the behavior and communication style of the simulated advisors.