# NearbySpiralsTEST

An exploratory, clearly quality-flagged presentation of the 35 galaxies in `Nearby_IMBH/JUST_THE_35_GALAXIES`. No source FITS or prior scientific result was overwritten.

## Inputs

35 512×512 FITS frames, 34 existing color references, saved AutoProf geometry where available, saved Gaia foreground-star masks where available, and saved exploratory Track-V2 and Fourier candidate overlays from BIRTHDAY_GITHUB_STYLE_V6. The sample combines DSS2 and Legacy survey images; flux units, PSFs and calibration are not uniform.

## Methods

- Track-V2: earlier candidate log-spiral traces reused as saved. No new Track-V2 detection run is claimed.
- Curved Gabor: structure-tensor direction field and local curved sampling to enhance ridges in the provisional smooth-subtracted image. Skeleton-connected fragments propose tracks; these are **not** validated physical arms, and disconnected branches can be misinterpreted.
- Fourier: saved exploratory Fourier components reused as saved. No multi-peak/multi-mode reanalysis is claimed.
- SpArcFiRe: genuine installed compiled r140 engine, single-turn configuration, run on AutoProf-deprojected image frames. Arcs are screened for valid pitch and extent. The engine may detect stellar clumps or spurious structure; these are exploratory, not confirmed arms.
- Bulge and disk: a new robust two-component **radial-median approximation**, exponential disk and fixed-index n=2 bulge-like profile, not GALFIT, not PSF-convolved, not a calibrated mass decomposition. Bar was **not fitted** and its panel is zero. No B/T should be inferred.
- Arm light: one nonnegative 2.5-native-pixel Gaussian ribbon per candidate path fitted to the original component-subtracted pixel values. This is in-sample and can overfit. No physical interpretation is assigned to amplitudes.

## Residual arithmetic

`before = (observed - estimated sky) - (provisional bulge + provisional disk + zero bar)`; `after = before - fitted arm light`. Both are signed numerical differences in FITS image units before visualization.

## Known limitations

Many AutoProf inclination estimates are unstable, foreground stars and star-forming clumps can be mistaken for arms, and original images are not uniformly calibrated. UGCA105 has no reliable AutoProf geometry in the saved analysis. High-inclination objects are unsuitable for precise pitch. M33, NGC300 and NGC7793 have published central black-hole constraints and should not be presented as unconstrained new IMBH candidates. This site is a method test, not a validated black-hole candidate catalog.

The source scripts are preserved separately on the authorized local computer at `E:/Astro/NearbySpiralsTEST_Work`. Raw FITS, original working catalogs, and personal filesystem contents are not published.

## Reproducibility
The source/ folder includes the actual exploratory analysis and publication scripts. They expect the FITS files and saved AutoProf/legacy products from the user's local Nearby_IMBH workspace; no raw FITS are redistributed. SpArcFiRe additionally uses the separately installed r140 binary. Paths are local-workstation defaults and must be adapted for another machine.

