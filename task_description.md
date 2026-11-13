## Common requirements
 
- **Take an assignment:** Choose a Homework. Write its #number in the Teams Homework Assignments channel.
One task can only be taken by one person. If it is accidentally taken by 2 people, the lecturers will have to give the second person the green light, or ask them to choose a different task.
- **Hand in:** By December 7th, at moodle -- https://edu.vik.bme.hu
- **Deliverables:** a public Git repo, a deployed URL (GitHub Pages, Vercel or Hugging Face Spaces), a README on usage, the experiences and lessons learned. (E.g. which copilots and agents were used, the key prompts, and what had to be fixed by hand.) Optional but also preferred: a 3-minute demo video as well.
- **Analysis element:** at least one ML, statistics or evaluation component the student can explain and defend at an oral defense. We will have one short question about your work at the oral defence, 1 minute answer, just a check, no bigggie.
(Note: the porpuse is that I choose a component in the system, and I can describe and defend why I did it in the way I did it. In my task it is not a ML, staticstics or evaluation component. In my task it is about the system design or the architectural decisions.)
- **Time budget:** about 15–20 hours. Grade against the required minimum, not against the amount of polish.
- **Checkpoint before Nov.20:** confirm data access works. Several sources need a free registration: OpenSky, ENTSO-E, OpenAQ, BKK FUTÁR, Cloudflare Radar, NASA FIRMS, Etherscan, AISStream and OpenCelliD.
- **Live data:** forecasting tasks (2, 3, 4, 21, 23, 26) need either a historical archive or 1–2 weeks of data collected by the student. Start logging early.

## Chosen task
**31. Interactive textbook chapter on Bayesian inference**
Write a chapter with runnable in-browser Python (JupyterLite or Pyodide). Cover priors, posteriors, conjugacy, MCMC and variational inference. Every concept needs one interactive figure and one exercise with an automatic check.