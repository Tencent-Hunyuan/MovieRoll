(() => {
const atmosphere = document.querySelector('#hero-atmosphere');
const particles = document.querySelector('#ambient-particles');
const header = document.querySelector('.paper-header');
const pageAtmosphere = document.querySelector('#page-atmosphere');
const siteBackdrop = document.querySelector('.site-backdrop');
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
let headerVisible = true;

// Fixed positions avoid layout changes or random flashes between reloads.
for (let i = 0; i < 12; i++) {
  const particle = document.createElement('span');
  particle.className = 'ambient-particle';
  const lane = i % 2 === 0 ? 2 : 77;
  particle.style.setProperty('--x', `${lane + ((i * 17) % 20)}%`);
  particle.style.setProperty('--y', `${90 + ((i * 89) % 900)}px`);
  particle.style.setProperty('--size', `${i % 11 === 0 ? 2.5 : i % 4 === 0 ? 1.5 : 1}px`);
  particle.style.setProperty('--duration', `${12 + (i % 7) * 2}s`);
  particle.style.setProperty('--delay', `${-(i % 11) * 2}s`);
  particles.append(particle);
}

// Build the decorative reels from the same catalog as the galleries.
// Interleave tasks/models so neighboring frames show different scenes.
async function renderReels() {
  const catalog = await window.loadMovieRollContent();
  const groups = Object.values(catalog.showcases).map(records => records.filter(record => record.poster));
  const samples = [];
  for (let i = 0; i < Math.max(...groups.map(group => group.length)); i++) {
    for (const group of groups) if (group[i]) samples.push(group[i]);
  }
  if (!samples.length) return;
  const alternate = [...samples.slice(Math.floor(samples.length / 2)), ...samples.slice(0, Math.floor(samples.length / 2))].reverse();
  const reels = document.querySelector('#rolling-reels');
  for (const [side, records] of [samples, alternate].entries()) {
    const reel = document.createElement('div');
    reel.className = `rolling-reel ${side === 0 ? 'left' : 'right'}`;
    reel.style.setProperty('--roll-duration', `${records.length * (side === 0 ? 4.25 : 5.125)}s`);
    reel.style.setProperty('--roll-delay', side === 0 ? '0s' : '-17s');
    const tape = document.createElement('div');
    tape.className = 'roll-tape';
    const cycle = document.createElement('div');
    cycle.className = 'roll-cycle';
    for (const record of records) {
      const frame = document.createElement('div');
      frame.className = 'roll-frame';
      frame.dataset.case = record.id;
      const image = document.createElement('img');
      image.src = record.poster;
      image.alt = '';
      image.decoding = 'async';
      image.draggable = false;
      image.width = 864;
      image.height = 480;
      frame.append(image);
      cycle.append(frame);
    }
    // The second copy is only for a seamless loop, not another catalog sample.
    tape.append(cycle, cycle.cloneNode(true));
    const window = document.createElement('div');
    window.className = 'roll-window';
    const scan = document.createElement('div');
    scan.className = 'roll-scan';
    window.append(scan);
    reel.append(tape, window);
    reels.append(reel);
  }
}
renderReels().catch(() => { document.querySelector('#rolling-reels').dataset.unavailable = 'true'; });

function updateMotion() {
  const paused = reducedMotion.matches;
  siteBackdrop.classList.toggle('is-paused', paused || document.hidden);
  const running = !paused && headerVisible && !document.hidden;
  atmosphere.classList.toggle('is-paused', !running);
  atmosphere.dataset.running = String(running);
  const pageRunning = !paused && !headerVisible && !document.hidden;
  pageAtmosphere.classList.toggle('is-visible', !headerVisible);
  pageAtmosphere.classList.toggle('is-paused', !pageRunning);
  pageAtmosphere.dataset.running = String(pageRunning);
  document.documentElement.classList.toggle('reduce-motion', paused);

}

reducedMotion.addEventListener('change', updateMotion);
document.addEventListener('visibilitychange', updateMotion);
new IntersectionObserver(entries => {
  headerVisible = entries[0].isIntersecting;
  updateMotion();
}, { rootMargin: '80px 0px' }).observe(header);
updateMotion();


// Reveal headings once, keeping their content readable throughout the transition.
const revealed = new WeakSet();
const headingSelector = '.section-heading, .group-heading, .comparison-heading, .method-block > h3';
const revealObserver = new IntersectionObserver(entries => {
  for (const entry of entries) if (entry.isIntersecting) {
    entry.target.classList.add('is-revealed');
    revealObserver.unobserve(entry.target);
  }
}, { threshold: .25, rootMargin: '0px 0px -20px' });
function registerHeadings() {
  for (const heading of document.querySelectorAll(headingSelector)) {
    if (revealed.has(heading)) continue;
    revealed.add(heading);
    heading.classList.add('reveal-ready');
    revealObserver.observe(heading);
  }
}
const contentObserver = new MutationObserver(() => {
  registerHeadings();
  if (document.querySelector('#method-figures [data-figure]')) contentObserver.disconnect();
});
contentObserver.observe(document.querySelector('#main'), { childList: true, subtree: true });
registerHeadings();
})();
