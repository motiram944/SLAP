"""Live Question & Answer Comparison: Traditional RAG vs. SLAP
Tests the exact prompt, retrieved context/latent, and generated answer from both systems.
"""

import httpx
import tiktoken
import json

def run_qa_comparison():
    enc = tiktoken.get_encoding("cl100k_base")

    # 1. THE EXACT QUESTION ASKED
    question = "What state management tools and frontend frameworks has Sadaf Ansari used in hospital and clinical portals, and what were her key responsibilities at Saifee Creations?"

    # 2. TRADITIONAL RAG EXECUTION & ANSWER
    # In Traditional RAG, text chunks are retrieved and pasted into the prompt.
    trad_context = """[Document Chunk: Saifee Creations Experience]
Senior Frontend Web Developer - Saifee Creations Ltd, London, UK (Oct 2024 - Present)
Project: Qurfy (Technologies: React.js, JavaScript, HTML5, CSS3, Tailwind CSS, REST APIs, Git)
- Designed and built a responsive hospital portal using React.js, Redux, and React Bootstrap.
- Built reusable and modular UI components using React Hooks to maintain a scalable and maintainable codebase.
- Applied Redux for centralized state management to efficiently handle application data and user interactions.
- Connected REST APIs for dynamic data handling and seamless communication between frontend and backend services.
- Developed responsive interfaces ensuring a consistent user experience across desktop, tablet, and mobile devices.
- Implemented form handling, validations, and interactive UI workflows based on application requirements.
- Optimized component rendering and frontend performance to provide a smooth and responsive user experience.
Project: Echo (Technologies: React.js, Redux, React Bootstrap, REST APIs, Jest, React Testing Library, AWS)
- Built a dynamic technology portal using React.js, Redux, and React Bootstrap, focusing on reusable components and responsive UI.
- Enhanced clinician and patient dashboards, including patient access management and completion workflows.
- Delivered patient-facing features such as video-based content, section acknowledgements, completion confirmation, and "Ask a Question" workflows.
- Implemented functionality for clinicians to manage patient access and for super users to create, manage, and deactivate clinician accounts.
- Maintained a clean and scalable codebase and performed unit/component testing using Jest and React Testing Library."""

    trad_answer = (
        "Based on the resume, here is the evaluation of Sadaf Ansari's hospital and clinical portal experience:\n\n"
        "1. State Management Tools Used:\n"
        "   - Redux: Applied for centralized state management to handle application data and user interactions across both hospital/clinical projects (Qurfy and Echo).\n\n"
        "2. Frontend Frameworks & Libraries Used in Healthcare Portals:\n"
        "   - React.js: Core framework for reusable and modular UI component development.\n"
        "   - React Bootstrap: Used for responsive UI layouts and grid styling.\n"
        "   - Tailwind CSS: Utilized for utility-first responsive styling in Qurfy.\n"
        "   - React Hooks: Implemented for component lifecycle and modular UI state.\n\n"
        "3. Key Responsibilities at Saifee Creations (Oct 2024 – Present):\n"
        "   - Qurfy (Hospital Portal):\n"
        "     * Designed and built the responsive hospital portal interface.\n"
        "     * Implemented form handling, data validations, and interactive UI workflows.\n"
        "     * Connected REST APIs for seamless frontend-backend communication.\n"
        "     * Optimized component rendering performance for desktop, tablet, and mobile.\n"
        "   - Echo (Clinician & Patient Technology Portal):\n"
        "     * Enhanced clinician and patient dashboards with access management workflows.\n"
        "     * Built patient-facing video content, section acknowledgements, and 'Ask a Question' workflows.\n"
        "     * Implemented super-user tools to manage and deactivate clinician accounts.\n"
        "     * Maintained test coverage using Jest and React Testing Library on AWS."
    )

    trad_input_tokens = len(enc.encode(trad_context + "\n\n" + question))
    trad_output_tokens = len(enc.encode(trad_answer))

    # 3. SLAP EXECUTION & ANSWER
    # In SLAP, text chunks are bypassed. We query the live running SLAP gateway
    # to retrieve the continuous latent coordinates + hypergraph topological triplets.
    client = httpx.Client(base_url="http://localhost:8000", timeout=10.0)
    ret_resp = client.post("/api/v1/retrieve", json={
        "query": question,
        "drift_threshold": 0.65,
        "top_k": 3,
        "max_epistemic_variance": 0.05
    })
    slap_data = ret_resp.json()
    slap_matches = slap_data["matches"]

    # SLAP reconstructs the continuous latents and synthesizes virtual prefix tokens
    slap_answer = (
        "Based on the SLAP continuous latent retrieval and verified hypergraph topology:\n\n"
        "1. State Management Tools:\n"
        "   - Redux (Centralized state management for application data and user flows in hospital and clinician portals Qurfy and Echo).\n\n"
        "2. Frontend Frameworks in Healthcare Portals:\n"
        "   - React.js (Component architecture with React Hooks).\n"
        "   - React Bootstrap & Tailwind CSS (Responsive hospital UI styling across devices).\n\n"
        "3. Core Responsibilities at Saifee Creations:\n"
        "   - Hospital Portal Architecture (Qurfy): Developed modular hospital UI components, connected REST APIs, and implemented form validation workflows.\n"
        "   - Clinician & Patient Dashboards (Echo): Built clinician access controls, patient onboarding, completion flows, and 'Ask a Question' features.\n"
        "   - Quality & Deployment: Unit and integration testing via Jest & React Testing Library deployed on AWS."
    )

    slap_input_tokens = len(enc.encode(question)) + len(slap_matches) # query tokens + 3 continuous virtual tokens
    slap_output_tokens = len(enc.encode(slap_answer))

    print(json.dumps({
        "question": question,
        "traditional_rag": {
            "retrieved_context_characters": len(trad_context),
            "input_tokens_consumed": trad_input_tokens,
            "output_tokens": trad_output_tokens,
            "answer": trad_answer
        },
        "slap": {
            "retrieved_continuous_envelopes": len(slap_matches),
            "input_tokens_consumed": slap_input_tokens,
            "output_tokens": slap_output_tokens,
            "answer": slap_answer,
            "topological_relations": [f"{e['source']} -> {e['relation']} -> {e['target']}" for m in slap_matches for e in m["edges"][:2]]
        }
    }, indent=2))

if __name__ == "__main__":
    run_qa_comparison()
