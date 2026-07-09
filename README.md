Verbatim is a Q&A Librarian for California Legislation. A user asks a question in plain english, and the system searches through 10,761 files (pertaining to code sections like Civil Code, Labor Code, and others) stored in a vector database. It then returns a cited quote of that exact text, with no paraphrasing or synthesis of a response, along with a source link. If no source is found within California Law, it recommends a specific external legal research database instead of guessing or giving no answer. 

This serves as a useful research tool for the students of Santa Clara University's BEACH Consulting program. Without AI hallucinations nor synthesis, it allows students to draw their own conclusions and inferences while analyzing sources. Furthermore, it drastically cuts down on research time, quickly providing information without the need to sift through thousands of documents for a single piece of information. 

For Local Setup: 
1) Create venv and set up requirements.txt
2) API Setup:
   cd backend
   touch .env
   fill the following variables:
     OPENAI_API_KEY
     AWS_ACCESS_KEY_ID
     AWS_SECRET_ACCESS_KEY
     AWS_DEFAULT_REGION
     OPENAI_API_KEY
     S3_BUCKET_NAME
3) In "backend" folder, run "python main.py", and let the vector data base pull from the S3 bucket and set up.
4) Open up the extension in Chrome, and you're good to go! 
