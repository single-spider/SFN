# Vision-Guided Peg-in-Hole Research Showcase

Research website for the SFN paper implementation, controller extensions, simulation results, and recorded Panda replays.

## Local development

- `npm ci` installs the locked dependencies.
- `npm run dev` starts the local Vite site.
- `npm run build` creates the Sites-compatible build.
- `npm run build:pages` creates a static GitHub Pages build.

## GitHub Pages

The main repository includes `.github/workflows/deploy-showcase-pages.yml`. After pushing it to GitHub:

1. Open **Settings → Pages** in the GitHub repository.
2. Set **Build and deployment → Source** to **GitHub Actions**.
3. Push to `main`, or run **Deploy research site to GitHub Pages** manually from the Actions tab.

The workflow obtains the repository base path from GitHub Pages, builds the Vite app, uploads `dist`, and deploys it. GitHub Pages builds use hash-based routes so refreshing a section such as `#/journey` does not produce a 404. Local and Sites deployments retain ordinary browser routes.

All robot results presented by the site are from dynamic PyBullet simulation, not physical-robot experiments.
