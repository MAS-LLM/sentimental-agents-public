TOPIC = """What are the pros and cons of hiring this candidate: {candidate_name}

(with this bio: 
-----START OF CANDIDATE RESUME/BIO-----
{candidate_bio}
-----END OF CANDIDATE RESUME/BIO-----
)

for this role: {role_to_fill}, 
given this role description: 
-----START OF JOB DESCRIPTION-----
{role_description}
-----END OF JOB DESCRIPTION-----

Evaluate this candidate's suitability for the role based on your professional expertise."""

ADVISOR_PRIORITIES = """Please reply with the main objectives and priorities of {name}, in {word_limit} words or less.
Speak directly to {name}.
Be concise and to the point.
Give them a point of view.
Do not add anything else."""

ADVISOR_DESCRIPTION = """Please reply with a description of {name}, in {word_limit} words or less. 
Speak directly to {name}.
Stick to the professional experience of {name}, and their expertise.
Give them a point of view.
Do not add anything else."""

ADVISOR_CRITERIA = """Please reply with a bullet point list of the main criteria that {name} should use to evaluate if a candidate is a good match 
for the role: {role_to_fill}.
These criteria should be specifically relevant to the {name} function, objectives and priorities.
These criteria should be original and specific to your role.
Don't exceed 5 bullet points or {word_limit} words. 
Speak directly to {name}.
Give them a point of view.
Do not add anything else."""

SYSTEM_MESSAGE = """{conversation_description}

Your name is {name}.
Speak in the first person from the perspective of {name}

Do not change roles!
Do not speak from the perspective of anyone else.

----------------------------------------------------------------
Your description is as follows: {description}

----------------------------------------------------------------
Your objectives and priorities are as follows: {priority}

----------------------------------------------------------------
Your evaluation criteria are as follow: {criterion}

----------------------------------------------------------------
Evaluate this candidate's suitability for the role based on your professional expertise.

Stop speaking the moment you finish your evaluation."""

SPECIFIC_TOPIC = """{topic}

You are the facilitator.
Please make the topic more specific for evaluation.
Please reply with the specified evaluation focus in {word_limit} words or less. 
Speak directly to the participants: {names}.
Do not add anything else.
"""

# ANALYTICS_TEMPLATE = '''
# Analyze the following discussion data and identify key conversation patterns.
#
# Your report should include:
# a. Points where agents expressed definitive positions (strong opinions)
# b. Areas where agents showed agreement or consensus (agreeable opinions)
# c. Topics that received substantial discussion or multiple mentions (extensively discussed opinions)
#
# ====DATA====
#
# {data}
# '''



























# TOPIC = """What are the pros and cons of hiring this candidate: {candidate_name}
#
# (with this bio:
#
# -----START OF CANDIDATE RESUME/BIO-----
# {candidate_bio}
# -----END OF CANDIDATE RESUME/BIO-----
# )
#
#
# for this role: {role_to_fill},
# given this role description:
#
# -----START OF JOB DESCRIPTION-----
# {role_description}.
# -----END OF JOB DESCRIPTION-----
#
# Evaluate this candidate's suitability for the role based on your professional expertise.
# """
#
# ADVISOR_PRIORITIES = """Please reply with the main objectives and priorities of {name}, in {word_limit} words or less.
# Speak directly to {name}.
# Be concise and to the point.
# Give them a point of view.
# Do not add anything else."""
#
# ADVISOR_DESCRIPTION = """Please reply with a description of {name}, in {word_limit} words or less.
# Speak directly to {name}.
# Stick to the professional experience of {name}, and their expertise.
# Give them a point of view.
# Do not add anything else."""
#
# ADVISOR_CRITERIA = """Please reply with a bullet point list of the main criteria that {name} should use to evaluate if a candidate is a good match
# for the role: {role_to_fill}.
# These criteria should be specifically relevant to the {name} function, objectives and priorities.
# These criteria should be original and specific to your role.
# Don't exceed 5 bullet points or {word_limit} words.
# Speak directly to {name}.
# Give them a point of view.
# Do not add anything else."""
#
# ADVISOR_PROFILE = """
# Please reply with a complete profile for each advisor.
# The profile should contain the following elements:
#
# Advisor Name: {name}
#
# Advisor Short bio: re-write the {description} at the first person. Make sure to tone it down, not sound too self-congratulatory and reduce the amount of superlatives. Stick to your professional experience and your expertise.
#
# Advisor objectives and priorities: re-write the {priority} at the first person. Avoid any corporate jargon. Be concise and to the point.
#
# Advisor evaluation criteria: {criterion}
#
#
# Do not add antyhing else.
# """
#
# #For describing your own body movements, wrap your description in '*'.
#
# SYSTEM_MESSAGE = """{conversation_description}
#
# Your name is {name}.
# Speak in the first person from the perspective of {name}
#
# Do not change roles!
# Do not speak from the perspective of anyone else.
#
# ----------------------------------------------------------------
# Your description is as follows: {description}
#
# ----------------------------------------------------------------
# Your objectives and priorities are as follows: {priority}
#
# ----------------------------------------------------------------
# Your evaluation criteria are as follow: {criterion}
#
# ----------------------------------------------------------------
# Evaluate the candidate based on your professional expertise and criteria.
# Your assessment should reflect your specific role and responsibilities.
# Provide your professional judgment on the candidate's fit for the position.
#
# Stop speaking the moment you finish your evaluation."""
#
#
#
# SYSTEM_MESSAGE2 = """
#
# Your name is {name}.
# Speak in the third person from the perspective of {name} starting with :'You are a {name}....'
#
# Do not change roles!
# Do not speak from the perspective of anyone else.
#
# ----------------------------------------------------------------
# Your description is as follows: {description}
#
#
# ----------------------------------------------------------------
# Your objectives and priorities are as follows: {priority}
#
#
# ----------------------------------------------------------------
# First state your evaluation criteria.
# These criteria should be specifically relevant to your name, description and your objectives and priorities.
# Your evaluation critera are as follow: {criterion}
#
#
# ----------------------------------------------------------------
# Your goal is to persuade your conversation partners of your point of view
# Your perspective should be strongly based on your criterias for evaluation.
# Your point of view should be original. Try to differetiate your point of view from your conversation partners.
# Avoid as much as possible generic statements that anyone could make.
#
#
# DO look up information with your tool to refute your partner's claims.
# DO cite your sources.
# DO NOT fabricate fake citations.
# DO NOT cite any source that you did not look up.
# Do not add anything else.
#
# Stop speaking the moment you finish speaking from your perspective.
# """
#
# SPECIFIC_TOPIC = """{topic}
#
# You are the facilitator.
# Please make the topic more specific.
# Please reply with the specified quest in {word_limit} words or less.
# Speak directly to the participants: {names}.
# Do not add anything else.
# """
#
#
# ANALYTICS_TEMPLATE = '''
# As an experienced Psychologist, your task is to analyze the following discussion data involving various agents.
#
# Your report should include:
# a. Identification of a maximum of 5 firm convictions expressed by the agents. (strong opinions)
# b. Highlighting a maximum of 5 viewpoints that are widely accepted or approved by the agents. (agreeable opinions)
# c. Pointing out a maximum of 5 subjects that have seen significant shifts in sentiment or perspective. (extensively discussed opinions)
#
# ====DATA====
#
# {data}
#
# '''