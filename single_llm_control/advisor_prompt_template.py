# ================== SINGLE LLM PROMPTS ==================

SINGLE_TOPIC = """What are the pros and cons of hiring this candidate: {candidate_name}

(with this bio: 
-----START OF CANDIDATE RESUME/BIO-----
{candidate_bio}
-----END OF CANDIDATE RESUME/BIO-----
)

for this role: {role_to_fill}, given this role description: 
-----START OF JOB DESCRIPTION-----
{role_description}
-----END OF JOB DESCRIPTION-----

Evaluate this candidate's suitability for the role based on your professional expertise."""

SINGLE_DESCRIPTION = """Please reply with a **description** of the UnifiedAdvisor, in {word_limit} words or less. 
Speak in the first person, stick to professional experience and expertise, and provide a clear point of view."""

SINGLE_PRIORITIES = """Please reply with the main **objectives and priorities** of the UnifiedAdvisor, in {word_limit} words or less. 
Be concise and provide a focused point of view."""

SINGLE_CRITERIA = """Please reply with a bullet list of the **evaluation criteria** the UnifiedAdvisor should use to judge if a candidate is a good match for this role: {role_to_fill}. 
Limit to 5 bullets or {word_limit} words. Be specific and relevant to the role."""

SINGLE_ADVISOR_PROMPT = """{topic}

{description}

{priorities}

{criteria}

Now, as the UnifiedAdvisor, evaluate this candidate's suitability for the role based on your professional expertise."""