# TICKET-016: Comic Font Selection and Packaging

## Summary
Select a legally appropriate open-source comic lettering font, download the `.ttf` file, place it in the `fonts/` directory, and update `config.yaml` with the path. No code is written in this ticket — it is a licensing and asset procurement step.

## Font Options and Licensing

| Font | License | Source | Notes |
|---|---|---|---|
| **Bancomics** | SIL OFL 1.1 | Google Fonts | Clean all-caps comic style, good readability at small sizes |
| **Noto Sans** | SIL OFL 1.1 | Google Fonts | Fallback option — not comic-style but fully readable and OFL |
| **Anime Ace 2.0** | Free personal use only | Blambot | Industry standard for manga localization; NOT for commercial redistribution |
| **CC Wild Words** | Free for non-commercial | Comicraft | Popular alternative; check license terms for your use case |

Recommendation:
- If this pipeline is for personal reading: **Anime Ace 2.0** is the most authentic choice
- If this pipeline output will be distributed: **Bancomics** (OFL) is the safe choice

## Steps

1. Download the chosen `.ttf` file
   - Bancomics: available via Google Fonts download
   - Anime Ace 2.0: download from blambot.com/pages/free-fonts (requires account)

2. Place the file at `fonts/anime_ace_2.ttf` (or `fonts/bancomics.ttf`)

3. Confirm it is readable by Pillow:
   ```bash
   uv run python -c "
   from PIL import ImageFont
   f = ImageFont.truetype('fonts/anime_ace_2.ttf', 16)
   print('Font loaded:', f.getname())
   "
   ```

4. Update `config.yaml`:
   ```yaml
   typesetting:
     font_path: "fonts/anime_ace_2.ttf"
   ```

5. Add the font filename pattern to `.gitignore` if the license prohibits redistribution:
   ```
   fonts/*.ttf
   ```
   (Users running this project must procure the font themselves if it is not freely redistributable)

## Bold and Regular Variants
Comic lettering typically uses all-caps with a bold weight for emphasis. Anime Ace 2.0 is inherently styled this way. If using a font with separate Regular/Bold files, Pillow handles them as separate font objects — you'd need both files and logic to switch. Avoid this complexity initially; use a font that is styled uniformly.

## Acceptance Criteria
- `fonts/<chosen_font>.ttf` exists in the project
- `config.yaml` references the correct path
- Pillow loads the font without error at sizes 8 through 36
- Font licensing status is documented in a `fonts/FONTS.txt` file with source URL and license type

## Dependencies
- TICKET-001 (fonts/ directory exists)
- TICKET-003 (config.yaml has typesetting section)

## Estimated Effort
1 hour
