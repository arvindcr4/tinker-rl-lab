-- Opens macOS Mail with pre-filled EAI submission draft and attachments.
-- Run: osascript /Users/arvind/tinker-rl-lab/paper/send_eai_email.applescript

set texPath  to POSIX file "/Users/arvind/tinker-rl-lab/paper/main_eai.tex"
set pdfPath  to POSIX file "/Users/arvind/tinker-rl-lab/paper/main_eai.pdf"
set bibPath  to POSIX file "/Users/arvind/tinker-rl-lab/paper/references.bib"

set emailSubject to "EAI Format Submission -- Plagiarism Check Request -- TinkerRL Audit"

set emailBody to "Dear PESU e-Library Team,

Please find attached our manuscript prepared in the EAI Endorsed Transactions (LNICST/Vancouver) format for submission.

Paper: Reward Contrast, Not Algorithm Labels: A Diagnostic Audit of Critic-Free Group-Relative RL for LLMs
Authors: Arvind C R, Sandhya Jeyaraj, Madhu Kumara L, Mohammad Rafi, Dhruva N Murthy, Arumugam K, Anwesh Reddy Paduri, Narayana Darapaneni
Length: 52 pages (incl. appendices) | Format: EAI Endorsed Transactions (single-column draft, Vancouver refs)

Request: Kindly perform a plagiarism check (Turnitin / iThenticate) before we submit to the EAI journal OJS portal. As per guidelines, plagiarism must be below 10%.

Attachments:
  1. main_eai.pdf -- compiled manuscript
  2. main_eai.tex -- LaTeX source
  3. references.bib -- BibTeX bibliography

CC: Mr. Anwesh Reddy Paduri (Great Learning), Ms. Sudha BG (Great Learning) -- for visibility.

Thank you for your support.

Best regards,
Arvind C R (on behalf of the authors)
PES University
arvindcr4@gmail.com"

tell application "Mail"
    activate
    set newMsg to make new outgoing message with properties {subject:emailSubject, content:emailBody, visible:true}
    tell newMsg
        make new to recipient at end of to recipients with properties {address:"pesu.eclib@pes.edu"}
        make new cc recipient at end of cc recipients with properties {address:"anwesh@greatlearning.in"}
        make new cc recipient at end of cc recipients with properties {address:"sudha.bg@greatlearning.in"}
        tell content
            make new attachment with properties {file name:pdfPath} at after last paragraph
            make new attachment with properties {file name:texPath} at after last paragraph
            make new attachment with properties {file name:bibPath} at after last paragraph
        end tell
    end tell
end tell
