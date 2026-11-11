const RAD2DEG = 180 / Math.PI;

function julianDate(unixTime) {
  return unixTime / 86400 + 2440587.5;
}

function localSiderealTimeDeg(unixTime, lonDeg) {
  const jd = julianDate(unixTime);
  const T = (jd - 2451545.0) / 36525.0;
  // Greenwich mean sidereal time (deg)
  let GMST = (280.46061837 + 360.98564736629 * (jd - 2451545.0)
    + 0.000387933 * T * T - (T * T * T) / 38710000) % 360;
  if (GMST < 0) GMST += 360;
  // Local sidereal time
  let LST = (GMST + lonDeg) % 360;
  if (LST < 0) LST += 360;
  return LST;
}

// ══════════════════════════════════════════════════════════════════════
//  TIME ZONES
//  Stored: Unix time (UTC) everywhere - databases, images/YYYYMMDD, files.
//  Shown: the station time zone (config.json "timezone", from
//  location_timezone in config.ini) or UTC, as chosen by the viewer
//  (setTimeMode, remembered in the browser). Intl converts every moment
//  with the time zone database, so summer/winter time is always right; a
//  fixed offset (timezone_offset) is never used for display.
// ══════════════════════════════════════════════════════════════════════
const TIME_MODE_KEY = 'allsky360.timeMode';
const STATION_TZ_KEY = 'allsky360.stationTimeZone';

function __lsGet(key) { try { return localStorage.getItem(key); } catch (_) { return null; } }
function __lsSet(key, value) { try { localStorage.setItem(key, value); } catch (_) {} }

function isValidTimeZone(tz) {
  if (!tz || typeof tz !== 'string') return false;
  try { new Intl.DateTimeFormat('en-GB', { timeZone: tz }); return true; } catch (_) { return false; }
}

// Last known station zone: available at once on the next page load
let __stationTimeZone = isValidTimeZone(__lsGet(STATION_TZ_KEY)) ? __lsGet(STATION_TZ_KEY) : null;

function getStationTimeZone() {
  return __stationTimeZone || Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';
}

function setStationTimeZone(tz) {
  if (!isValidTimeZone(tz) || tz === __stationTimeZone) return;
  __stationTimeZone = tz;
  __lsSet(STATION_TZ_KEY, tz);
  window.dispatchEvent(new CustomEvent('allsky-timezone-change'));
}

// 'local' (station time zone, default) or 'utc'
function getTimeMode() { return __lsGet(TIME_MODE_KEY) === 'utc' ? 'utc' : 'local'; }

function setTimeMode(mode) {
  __lsSet(TIME_MODE_KEY, mode === 'utc' ? 'utc' : 'local');
  window.dispatchEvent(new CustomEvent('allsky-timezone-change'));
}

function getDisplayTimeZone() { return getTimeMode() === 'utc' ? 'UTC' : getStationTimeZone(); }

const __dtfCache = new Map();
function __dtf(tz) {
  let f = __dtfCache.get(tz);
  if (!f) {
    f = new Intl.DateTimeFormat('en-GB', {
      timeZone: tz, year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23',
      timeZoneName: 'short'
    });
    __dtfCache.set(tz, f);
  }
  return f;
}

// Calendar fields of a moment in a time zone, plus the zone abbreviation
// (CET, CEST, UTC, ...)
function zonedParts(unixSec, tz = getDisplayTimeZone()) {
  const out = {};
  for (const p of __dtf(tz).formatToParts(new Date(unixSec * 1000)))
    out[p.type] = p.value;
  return {
    year: +out.year, month: +out.month, day: +out.day,
    hour: +out.hour % 24, minute: +out.minute, second: +out.second,
    zone: out.timeZoneName || tz
  };
}

const __pad2 = (n) => String(n).padStart(2, '0');

// "2026-11-03 11:29:41 CET"
function formatDateTime(unixSec, { tz = getDisplayTimeZone(), seconds = true, zone = true } = {}) {
  const p = zonedParts(unixSec, tz);
  let s = `${p.year}-${__pad2(p.month)}-${__pad2(p.day)} ${__pad2(p.hour)}:${__pad2(p.minute)}`;
  if (seconds) s += `:${__pad2(p.second)}`;
  return zone ? `${s} ${p.zone}` : s;
}

// "11:29" (optionally with seconds and zone)
function formatClock(unixSec, { tz = getDisplayTimeZone(), seconds = false, zone = false } = {}) {
  const p = zonedParts(unixSec, tz);
  let s = `${__pad2(p.hour)}:${__pad2(p.minute)}`;
  if (seconds) s += `:${__pad2(p.second)}`;
  return zone ? `${s} ${p.zone}` : s;
}

// Zone abbreviation valid at a moment (CET in winter, CEST in summer)
function timeZoneLabel(unixSec = Date.now() / 1000, tz = getDisplayTimeZone()) {
  return zonedParts(unixSec, tz).zone;
}

// UTC offset of a time zone at a moment, in seconds
function zoneOffsetSeconds(unixSec, tz = getDisplayTimeZone()) {
  const t = Math.floor(unixSec);
  const p = zonedParts(t, tz);
  return Date.UTC(p.year, p.month - 1, p.day, p.hour, p.minute, p.second) / 1000 - t;
}

// Unix time of 00:00 local time on a calendar day (DST-safe: the offset is
// taken at the result, not at the guess)
function localMidnight(year, month, day, tz = getDisplayTimeZone()) {
  const asUtc = Date.UTC(year, month - 1, day) / 1000;
  let t = asUtc - zoneOffsetSeconds(asUtc, tz);
  t = asUtc - zoneOffsetSeconds(t, tz);
  return t;
}

function startOfDayInZone(unixSec, tz = getDisplayTimeZone()) {
  const p = zonedParts(unixSec, tz);
  return localMidnight(p.year, p.month, p.day, tz);
}

// "YYYY-MM-DD" of a moment in a time zone
function dayStringInZone(unixSec = Date.now() / 1000, tz = getDisplayTimeZone()) {
  const p = zonedParts(unixSec, tz);
  return `${p.year}-${__pad2(p.month)}-${__pad2(p.day)}`;
}

// [start, end) of a calendar day in a time zone, Unix seconds (23 or 25 h
// on the days of the summer/winter time change)
function dayRangeInZone(dayStr, tz = getDisplayTimeZone()) {
  const [y, m, d] = dayStr.split('-').map(Number);
  const next = new Date(Date.UTC(y, m - 1, d + 1));
  return {
    start: localMidnight(y, m, d, tz),
    end: localMidnight(next.getUTCFullYear(), next.getUTCMonth() + 1, next.getUTCDate(), tz)
  };
}

// UTC days ("YYYY-MM-DD") touched by [start, end): the daily databases and
// image folders are UTC days, a local day can span two of them
function utcDaysForRange(start, end) {
  const days = [];
  for (let t = Math.floor(start / 86400) * 86400; t < end; t += 86400)
    days.push(new Date(t * 1000).toISOString().slice(0, 10));
  return days;
}

// Rows of a day-based API endpoint (e.g. /api/meteo) for a calendar day in
// the display time zone: fetches the UTC days it spans, keeps its range.
async function fetchDayRows(endpoint, dayStr, tz = getDisplayTimeZone()) {
  const { start, end } = dayRangeInZone(dayStr, tz);
  const parts = await Promise.all(utcDaysForRange(start, end).map(async day => {
    try {
      const res = await fetch(`${endpoint}?date=${day}`);
      if (!res.ok) return [];
      const json = await res.json();
      return Array.isArray(json) ? json : [];
    } catch (err) {
      console.error(`Failed to load ${endpoint} for ${day}:`, err);
      return [];
    }
  }));
  return parts.flat()
    .filter(r => r.timestamp >= start && r.timestamp < end)
    .sort((a, b) => a.timestamp - b.timestamp);
}

// Chart.js time axes: tick labels and tooltip titles in the display time
// zone (the date adapter would use the browser's zone)
function chartTimeTicks(extra = {}) {
  return { ...extra, callback: (value) => formatClock(value / 1000) };
}
function chartTooltipTitle(items) {
  return items.length ? formatDateTime(items[0].parsed.x / 1000, { seconds: false }) : '';
}

// ── Line icons (lucide, 1.5 stroke) for labels and chips ───────────────
const ICONS = {
  moon:   '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>',
  sun:    '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>',
  clock:  '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
  lock:   '<rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
  unlock: '<rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 9.9-1"/>',
  pin:    '<path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/>',
  reset:  '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>',
  reload: '<path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/>',
  star:   '<path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>'
};
function icon(name) {
  return `<svg class="ico" viewBox="0 0 24 24" aria-hidden="true">${ICONS[name] || ''}</svg>`;
}
// "Night" / "Day" with its icon (innerHTML)
function dayNightLabel(night) {
  return `${icon(night ? 'moon' : 'sun')} ${night ? 'Night' : 'Day'}`;
}

// Button that switches between station time and UTC; label "Local · CET"
function initTimeModeToggle(el) {
  if (!el) return;
  const render = () => {
    const utc = getTimeMode() === 'utc';
    el.innerHTML = icon('clock') + (utc ? ' UTC' : ` Local · ${timeZoneLabel(Date.now() / 1000, getStationTimeZone())}`);
    el.title = utc
      ? 'Times are shown in UTC. Click for the local time of the station.'
      : `Times are shown in the station time zone ${getStationTimeZone()}. Click for UTC.`;
  };
  el.addEventListener('click', () => setTimeMode(getTimeMode() === 'utc' ? 'local' : 'utc'));
  window.addEventListener('allsky-timezone-change', render);
  render();
}

// Kept for the pages: the offset argument is ignored, the zone comes from
// the station configuration and the viewer's choice.
// ── Icon navigation (left bar), shared by all pages ───────────────────
// One place for the page list: a page only has <nav id="icon-nav"></nav>
// followed by <script>renderIconNav();</script> (synchronous, no flicker).
// null is a separator between the groups.
const __NAV_SVG = (body) =>
  `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${body}</svg>`;
const NAV_ITEMS = [
  { href: 'index.html', title: 'Home', icon: '<path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline>' },
  { href: 'dashboard.html', title: 'Dashboard', icon: '<rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect>' },
  null,
  { href: 'archive.html', title: 'Archive', icon: '<polyline points="21 8 21 21 3 21 3 8"></polyline><rect x="1" y="3" width="22" height="5"></rect><line x1="10" y1="12" x2="14" y2="12"></line>' },
  { href: 'videos.html', title: 'Videos', icon: '<polygon points="23 7 16 12 23 17 23 7"></polygon><rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect>' },
  { href: 'night.html', title: 'Night', icon: '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>' },
  null,
  { href: 'meteo.html', title: 'Weather', icon: '<path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"></path>' },
  { href: 'aurora.html', title: 'Aurora', icon: '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>' },
  { href: 'aircraft.html', title: 'Aircraft', icon: '<path d="M21 16l-4-4V6a2 2 0 0 0-4 0v6l-4 4H3l2 3h5l2-1 2 1h5l2-3z"></path>' },
  null,
  { href: 'graphs.html', title: 'Camera', match: ['graphs.html', 'hdr.html', 'processing.html', 'calibration.html'], icon: '<path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle>' },
  { href: 'status.html', title: 'System Status', health: true, icon: '<rect x="2" y="3" width="20" height="7" rx="2"></rect><rect x="2" y="14" width="20" height="7" rx="2"></rect><line x1="6" y1="6.5" x2="6.01" y2="6.5"></line><line x1="6" y1="17.5" x2="6.01" y2="17.5"></line>' },
  { href: 'config.html', title: 'Config', icon: '<circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"></path>' }
];

// File name of the current page; the API serves dashboard.html for /
function __currentPage() {
  return location.pathname.split('/').pop() || 'dashboard.html';
}

// An entry is active on its own page and on the pages listed in match
// (the camera entry stands for all camera tabs)
function __navActive(item, page) {
  return item.href === page || (item.match || []).includes(page);
}

function renderIconNav(nav = document.getElementById('icon-nav')) {
  if (!nav) return;
  const page = __currentPage();
  nav.innerHTML = NAV_ITEMS.map(item => item === null
    ? '<div class="nav-sep"></div>'
    : `<a href="${item.href}" class="nav-item${__navActive(item, page) ? ' active' : ''}"${item.health ? ' id="nav-health"' : ''} title="${item.title}" aria-label="${item.title}"${__navActive(item, page) ? ' aria-current="page"' : ''}>${__NAV_SVG(item.icon)}</a>`
  ).join('');
  __startNavHealth();
}

// Warning/error dot on the System Status icon on every page: polls
// api/system/status once a minute (the API caches it for a few seconds)
let __navHealthTimer = null;
function setNavHealth(status) {
  const el = document.getElementById('nav-health');
  if (!el) return;
  const problems = (status?.checks || []).filter(c => c.level === 'warn' || c.level === 'error');
  if (status && (status.overall === 'warn' || status.overall === 'error')) {
    el.dataset.health = status.overall;
    el.title = 'System Status · ' + problems.map(c => `${c.label}: ${c.text}`).join(' · ');
  } else {
    delete el.dataset.health;
    el.title = status ? 'System Status · all OK' : 'System Status';
  }
  el.setAttribute('aria-label', el.title);
}
async function __pollNavHealth() {
  try {
    const res = await fetch('api/system/status', { cache: 'no-store' });
    setNavHealth(res.ok ? await res.json() : null);
  } catch { setNavHealth(null); }
}
function __startNavHealth() {
  if (__navHealthTimer !== null || !document.getElementById('nav-health')) return;
  __navHealthTimer = setInterval(__pollNavHealth, 60000);
  __pollNavHealth();
}

// Tabs of the camera pages (one "Camera" entry in the icon navigation).
// A page has <nav class="sub-tabs" id="cam-tabs"></nav> in its header,
// followed by <script>renderCameraTabs();</script>
const CAMERA_TABS = [
  { href: 'graphs.html', title: 'Statistics' },
  { href: 'hdr.html', title: 'HDR' },
  { href: 'processing.html', title: 'Pipeline' },
  { href: 'calibration.html', title: 'Calibration' }
];
function renderCameraTabs(el = document.getElementById('cam-tabs')) {
  if (!el) return;
  const page = __currentPage();
  el.setAttribute('aria-label', 'Camera pages');
  el.innerHTML = CAMERA_TABS.map(t =>
    `<a href="${t.href}" class="sub-tab${t.href === page ? ' active' : ''}"${t.href === page ? ' aria-current="page"' : ''}>${t.title}</a>`
  ).join('');
}

// Text menu of the older pages (processing.html, panorama.html) from the
// same page list; extra: [{href, title}] pages only these menus show
function renderTextNav(list, extra = []) {
  if (!list) return;
  const page = __currentPage();
  list.innerHTML = [...NAV_ITEMS.filter(Boolean), ...extra].map(item =>
    `<li><a href="${item.href}" class="nav-link${__navActive(item, page) ? ' active' : ''}">${item.title}</a></li>`
  ).join('');
}

function formatTimestamp(unixTime, _timezoneOffsetSeconds = null) {
  return formatDateTime(unixTime);
}

// IAU 1976/2006 mean obliquity (arcsec poly), good enough for RA/Dec conversion
function meanObliquityRad(jd) {
  const T = (jd - 2451545.0) / 36525;
  const sec = 21.448 - 46.8150 * T - 0.00059 * T * T + 0.001813 * T * T * T; // ″
  const eps = (23 + 26 / 60 + sec / 3600) * Math.PI / 180;
  return eps;
}

const brightestStarsPoi = [
  { name: "Sirius", ra: 101.2875, dec: -16.7161, color: "yellow" },
  { name: "Canopus", ra: 95.9879, dec: -52.6957, color: "yellow" },
  { name: "Arcturus", ra: 213.9153, dec: 19.1824, color: "yellow" },
  { name: "Alpha Centauri", ra: 219.9021, dec: -60.8356, color: "yellow" },
  { name: "Vega", ra: 279.2347, dec: 38.7837, color: "yellow" },
  { name: "Capella", ra: 79.1723, dec: 45.9979, color: "yellow" },
  { name: "Rigel", ra: 78.6345, dec: -8.2056, color: "yellow" },
  { name: "Procyon", ra: 114.8255, dec: 5.2249, color: "yellow" },
  { name: "Achernar", ra: 24.4286, dec: -57.2368, color: "yellow" },
  { name: "Betelgeuse", ra: 88.7929, dec: 7.4071, color: "yellow" },
  { name: "Hadar (Beta Cen)", ra: 210.9558, dec: -60.3730, color: "yellow" },
  { name: "Altair", ra: 297.6958, dec: 8.8683, color: "yellow" },
  { name: "Acrux", ra: 186.6496, dec: -63.0991, color: "yellow" },
  { name: "Aldebaran", ra: 68.9800, dec: 16.5093, color: "yellow" },
  { name: "Antares", ra: 247.3519, dec: -26.4320, color: "yellow" },
  { name: "Spica", ra: 201.2983, dec: -11.1613, color: "yellow" },
  { name: "Pollux", ra: 116.3289, dec: 28.0262, color: "yellow" },
  { name: "Fomalhaut", ra: 344.4128, dec: -29.6222, color: "yellow" },
  { name: "Deneb", ra: 310.3579, dec: 45.2803, color: "yellow" },
  { name: "Regulus", ra: 152.0929, dec: 11.9672, color: "yellow" }
];

function getSymbol(night) {
  return dayNightLabel(night);
}

function cloudCoverToRGB(percent) {
  const colorStops = [
    { percent: 0, color: [0, 0, 60] },    // #00003C
    { percent: 30, color: [72, 72, 102] },   // #484866
    { percent: 40, color: [96, 96, 116] },   // #606074
    { percent: 70, color: [168, 168, 158] },  // #A8A89E
    { percent: 80, color: [192, 192, 172] },  // #C0C0AC
    { percent: 100, color: [240, 240, 200] }   // #F0F0C8
  ];

  // Eingabewert begrenzen
  percent = Math.max(0, Math.min(100, percent));

  // Passende Intervallgrenzen finden
  for (let i = 0; i < colorStops.length - 1; i++) {
    const a = colorStops[i];
    const b = colorStops[i + 1];

    if (percent >= a.percent && percent <= b.percent) {
      const t = (percent - a.percent) / (b.percent - a.percent);

      const r = Math.round(a.color[0] + (b.color[0] - a.color[0]) * t);
      const g = Math.round(a.color[1] + (b.color[1] - a.color[1]) * t);
      const bVal = Math.round(a.color[2] + (b.color[2] - a.color[2]) * t);

      return `rgb(${r}, ${g}, ${bVal})`;
    }
  }

  // Fallback
  return "rgb(240,240,200)";
}

// Number with fixed digits, or a dash for missing values (null/undefined/NaN):
// the aurora tool writes null when NOAA has no current value.
function fmtNum(value, digits = 1, fallback = '–') {
  return (typeof value === 'number' && Number.isFinite(value)) ? value.toFixed(digits) : fallback;
}

// ── Forecast helpers (forecast_meteo.json; meteo.html and dashboard.html) ──
// An hour counts as clear sky up to this total cloud cover (%)
const FORECAST_CLEAR_MAX = 25;
// Clear stretches shorter than this (hours) are not listed
const FORECAST_CLEAR_MIN_HOURS = 2;
// Precipitation probability (%) from which rain is shown as likely
const FORECAST_RAIN_LIKELY = 50;
// Dew risk when the air is less than this above the dew point (K)
const FORECAST_DEW_SPREAD_WARN = 2;
// Gusts from this speed are highlighted (km/h)
const FORECAST_GUST_WARN = 40;

// Contiguous runs [first, last] of indices where pred(hours[i]) holds
function hourRuns(hours, pred) {
  const out = [];
  let start = -1;
  hours.forEach((h, i) => {
    if (pred(h)) { if (start < 0) start = i; }
    else if (start >= 0) { out.push([start, i - 1]); start = -1; }
  });
  if (start >= 0) out.push([start, hours.length - 1]);
  return out;
}

// "21:00–04:00 (7 h)": end is the end of the last hour
function hourSpanText(hours, [a, b]) {
  return `${formatClock(hours[a].timestamp)}–${formatClock(hours[b].timestamp + 3600)} (${b - a + 1} h)`;
}

// The coming night of an hourly forecast: night = first run of dark hours
// (is_day 0) as [first, last] in hours, nightHours = those hours, clear =
// clear stretches of at least FORECAST_CLEAR_MIN_HOURS within nightHours,
// dewHours = night hours close to the dew point, rain = first run of likely
// rain in hours, maxGust = highest gust (km/h) of all hours
function forecastTonight(hours) {
  const num = (v) => typeof v === 'number' && Number.isFinite(v);
  const night = hourRuns(hours, h => h.is_day === 0)[0] || null;
  const nightHours = night ? hours.slice(night[0], night[1] + 1) : [];
  const clear = hourRuns(nightHours, h => num(h.cloud_cover) && h.cloud_cover <= FORECAST_CLEAR_MAX)
    .filter(([a, b]) => b - a + 1 >= FORECAST_CLEAR_MIN_HOURS);
  const dewHours = nightHours.filter(h => num(h.temperature) && num(h.dew_point) &&
                                         h.temperature - h.dew_point < FORECAST_DEW_SPREAD_WARN);
  const rain = hourRuns(hours, h => num(h.precipitation_probability) &&
                                    h.precipitation_probability >= FORECAST_RAIN_LIKELY)[0] || null;
  const maxGust = hours.reduce((m, h) => num(h.wind_gusts_10m) && h.wind_gusts_10m > m ? h.wind_gusts_10m : m, 0);
  return { night, nightHours, clear, dewHours, rain, maxGust };
}

// Colour of the jetstream speed (km/h) for the seeing label and the forecast
function getSeeingColor(jetstream_kmh) {
  if (jetstream_kmh <= 30) return "green";
  if (jetstream_kmh <= 60) return "limegreen";
  if (jetstream_kmh <= 90) return "orange";
  if (jetstream_kmh <= 120) return "orangered";
  return "red";
}

function getSeeingLabel(jetstream_kmh) {
  let label;
  if (jetstream_kmh <= 30) label = "excellent";
  else if (jetstream_kmh <= 60) label = "good";
  else if (jetstream_kmh <= 90) label = "fair";
  else if (jetstream_kmh <= 120) label = "poor";
  else label = "very poor";
  return `<span style="color: ${getSeeingColor(jetstream_kmh)}">(${label})</span>`;
}

function raDecToAzAlt(raDeg, decDeg, latDeg, lonDeg, unixTime) {
  const deg2rad = Math.PI / 180;
  const rad2deg = 180 / Math.PI;

  const jd = unixTime / 86400 + 2440587.5;
  const T = (jd - 2451545.0) / 36525.0;

  const GST = (280.46061837 + 360.98564736629 * (jd - 2451545.0)
    + 0.000387933 * T * T - T * T * T / 38710000) % 360;
  const LST = (GST + lonDeg + 360) % 360;

  const H = (LST - raDeg + 360) % 360;

  const hRad = H * deg2rad;
  const decRad = decDeg * deg2rad;
  const latRad = latDeg * deg2rad;

  const sinAlt = Math.sin(decRad) * Math.sin(latRad) + Math.cos(decRad) * Math.cos(latRad) * Math.cos(hRad);
  const altRad = Math.asin(sinAlt);

  const cosAz = (Math.sin(decRad) - Math.sin(altRad) * Math.sin(latRad)) / (Math.cos(altRad) * Math.cos(latRad));
  let azRad = Math.acos(Math.min(1, Math.max(-1, cosAz)));
  if (Math.sin(hRad) > 0) azRad = 2 * Math.PI - azRad;

  return {
    azimuth: azRad * rad2deg,
    altitude: altRad * rad2deg
  };
}

/* Image pixel <-> overlay canvas position for an <img> with
 * object-fit: contain, shown with translate(panX, panY) scale(zoomFactor)
 * around its centre (transform-origin: center), and a canvas covering the
 * untransformed <img> box. Uses the page globals panX, panY, zoomFactor. */
function imageFitScale(img) {
  const nw = img.naturalWidth || 1, nh = img.naturalHeight || 1;
  return Math.min(img.clientWidth / nw, img.clientHeight / nh);
}

function imageToCanvas(img, x, y) {
  const cw = img.clientWidth, ch = img.clientHeight;
  const fit = imageFitScale(img);
  const offX = (cw - (img.naturalWidth || 1) * fit) / 2;
  const offY = (ch - (img.naturalHeight || 1) * fit) / 2;
  return {
    x: cw / 2 + panX + zoomFactor * (offX + x * fit - cw / 2),
    y: ch / 2 + panY + zoomFactor * (offY + y * fit - ch / 2),
  };
}

function canvasToImage(img, cx, cy) {
  const cw = img.clientWidth, ch = img.clientHeight;
  const fit = imageFitScale(img) || 1;
  const offX = (cw - (img.naturalWidth || 1) * fit) / 2;
  const offY = (ch - (img.naturalHeight || 1) * fit) / 2;
  return {
    x: ((cx - cw / 2 - panX) / zoomFactor + cw / 2 - offX) / fit,
    y: ((cy - ch / 2 - panY) / zoomFactor + ch / 2 - offY) / fit,
  };
}

function azAltToImageXY(azimuthDeg, altitudeDeg) {
  azimuthDeg = (360 - azimuthDeg) % 360;

  let r = crosshairRadius * imageFitScale(imageEl) * zoomFactor;
  if (projectionType === "sinus") {
    r *= Math.sin((90 - altitudeDeg) * Math.PI / 180);
  } else {
    r *= (90 - altitudeDeg) / 90;
  }

  const angleRad = (azimuthDeg + crosshairRotation) * Math.PI / 180;
  const c = imageToCanvas(imageEl, crosshairCenterX, crosshairCenterY);

  const x = c.x + Math.sin(angleRad) * r;
  const y = c.y - Math.cos(angleRad) * r;

  return { x, y };
}

// Az/Alt -> RA/Dec  (Az von N über E, Alt über Horizont; alle Winkel in Grad)
function azAltToRaDec(latDeg, lonDeg, azDeg, altDeg, unixTime) {
  const d2r = Math.PI / 180, r2d = 180 / Math.PI;

  const φ = latDeg * d2r;
  const A = azDeg * d2r;
  const a = altDeg * d2r;

  // Deklination
  const sinδ = Math.sin(a) * Math.sin(φ) + Math.cos(a) * Math.cos(φ) * Math.cos(A);
  const δ = Math.asin(Math.min(1, Math.max(-1, sinδ)));

  // Stundenwinkel H über atan2 (sicheres Quadranten-Handling)
  const sinH = -Math.sin(A) * Math.cos(a) / Math.cos(δ);
  const cosH = (Math.sin(a) - Math.sin(φ) * Math.sin(δ)) / (Math.cos(φ) * Math.cos(δ));
  let H = Math.atan2(sinH, Math.min(1, Math.max(-1, cosH))); // rad

  // RA = LST - H
  let Hdeg = (H * r2d);
  if (Hdeg < 0) Hdeg += 360;

  let RA = localSiderealTimeDeg(unixTime, lonDeg) - Hdeg; // deg
  RA = ((RA % 360) + 360) % 360;

  const Dec = δ * r2d; // deg
  return { raDeg: RA, decDeg: Dec };
}

function formatRaHMS(raDeg) {
  const totalHours = raDeg / 15;
  const h = Math.floor(totalHours);
  const m = Math.floor((totalHours - h) * 60);
  const s = Math.round((((totalHours - h) * 60) - m) * 60);
  const pad = (n, w = 2) => String(n).padStart(w, "0");
  return `${pad(h)}:${pad(m)}:${pad(s)}`;
}

function formatDecDMS(decDeg) {
  const sign = decDeg < 0 ? "-" : "+";
  const a = Math.abs(decDeg);
  const d = Math.floor(a);
  const m = Math.floor((a - d) * 60);
  const s = Math.round((((a - d) * 60) - m) * 60);
  const pad = (n, w = 2) => String(n).padStart(w, "0");
  return `${sign}${pad(d)}°${pad(m)}′${pad(s)}″`;
}

function drawEquatorialPOI(ctx, name, raDeg, decDeg, latitude, longitude, color, pxRadius = 10) {
  const { azimuth, altitude } = raDecToAzAlt(raDeg, decDeg, latitude, longitude, imageTimestamp);
  drawHorizontalPOI(ctx, name, azimuth, altitude, color, pxRadius);
}

// Marker with label at azimuth/altitude (deg); nothing below the horizon
function drawHorizontalPOI(ctx, name, azimuth, altitude, color, pxRadius = 10) {
  if (altitude < 0) return;

  const { x, y } = azAltToImageXY(azimuth, altitude);
  ctx.save();
  ctx.beginPath();
  ctx.arc(x, y, pxRadius, 0, 2 * Math.PI);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();

  ctx.fillStyle = color;
  ctx.font = "11px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.fillText(name, x, y - pxRadius - 4);
  ctx.restore();
}

// robust root (some builds put exports under .default)
function Aroot() {
  if (window.Astronomia?.solar) return window.Astronomia;
  if (window.Astronomia?.default?.solar) return window.Astronomia.default;
  throw new Error("Astronomia bundle not available");
}

function astronomiaAvailable() {
  return !!(window.Astronomia?.solar || window.Astronomia?.default?.solar);
}

// The bundle is loaded with "defer" (or from the CDN as fallback, inserted
// later), so a page script that starts right away can run before it is
// there. Resolves once it is available; rejects after timeoutMs.
let astronomiaPromise = null;
function astronomiaReady(timeoutMs = 15000) {
  if (!astronomiaPromise) {
    astronomiaPromise = new Promise((resolve, reject) => {
      const t0 = Date.now();
      (function poll() {
        if (astronomiaAvailable()) resolve();
        else if (Date.now() - t0 > timeoutMs) {
          astronomiaPromise = null;
          reject(new Error("Astronomia bundle not available"));
        } else setTimeout(poll, 50);
      })();
    });
  }
  return astronomiaPromise;
}


function getSunRaDec(unixTime) {
  const A = Aroot();
  const jd = julianDate(unixTime);
  // already apparent equatorial (ra, dec in radians)
  const eq = A.solar.apparentEquatorial(jd);
  return { ra: eq.ra * RAD2DEG, dec: eq.dec * RAD2DEG };
}

function getMoonRaDec(unixTime) {
  const A = Aroot();
  const jd = julianDate(unixTime);

  // ecliptic coords (λ, β) in radians
  const m = A.moonposition.position(jd);
  const lambda = m.lon; // ecliptic longitude
  const beta = m.lat; // ecliptic latitude

  // convert ecliptic → equatorial
  const eps = meanObliquityRad(jd);
  const sinλ = Math.sin(lambda), cosλ = Math.cos(lambda);
  const tanβ = Math.tan(beta);

  const raRad = Math.atan2(sinλ * Math.cos(eps) - tanβ * Math.sin(eps), cosλ);
  const decRad = Math.asin(Math.sin(beta) * Math.cos(eps) + Math.cos(beta) * Math.sin(eps) * sinλ);

  // normalize RA to [0, 360)
  let ra = raRad * RAD2DEG;
  if (ra < 0) ra += 360;

  // distance Earth-Moon centres (km), for the parallax
  return { ra, dec: decRad * RAD2DEG, distanceKm: m.range };
}

// ── Horizontal positions of Sun and Moon for the station ───────────────
// Earth's equatorial radius (km), for the Moon's horizontal parallax
const EARTH_RADIUS_KM = 6378.14;
// Altitude of the centre at rise and set: upper limb on the horizon with
// standard refraction (34' + 16' semi-diameter), as in almanacs; for the
// Moon of its topocentric centre (its semi-diameter is about the Sun's)
const RISE_SET_ALTITUDE = -0.833;

// Sun: azimuth (deg from north over east) and geometric altitude (deg);
// the solar parallax (9") is negligible
function getSunAzAlt(unixTime, latDeg, lonDeg) {
  const sun = getSunRaDec(unixTime);
  return raDecToAzAlt(sun.ra, sun.dec, latDeg, lonDeg, unixTime);
}

// Moon: topocentric azimuth and geometric altitude. Seen from the Earth's
// surface instead of its centre the Moon is lower by its parallax
// P * cos(altitude), P = asin(R / distance), close to 1 deg (Meeus ch. 40)
function getMoonAzAlt(unixTime, latDeg, lonDeg) {
  const moon = getMoonRaDec(unixTime);
  const pos = raDecToAzAlt(moon.ra, moon.dec, latDeg, lonDeg, unixTime);
  const P = Math.asin(EARTH_RADIUS_KM / moon.distanceKm) * RAD2DEG;
  pos.altitude -= P * Math.cos(pos.altitude / RAD2DEG);
  return pos;
}

// Apparent altitude: geometric altitude plus atmospheric refraction
// (Saemundsson's formula, standard pressure and temperature), what the
// camera sees. Below -1 deg it no longer applies and is left out.
function apparentAltitude(altDeg) {
  if (altDeg < -1) return altDeg;
  const R = 1.02 / Math.tan((altDeg + 10.3 / (altDeg + 5.11)) / RAD2DEG); // arcmin
  return altDeg + R / 60;
}

// Altitude function of a body for the rise/set searches
function bodyAltitudeFn(body, latDeg, lonDeg) {
  const fn = body === 'moon' ? getMoonAzAlt : getSunAzAlt;
  return (t) => fn(t, latDeg, lonDeg).altitude;
}

// First crossing of RISE_SET_ALTITUDE between startUnix and endUnix:
// 5-minute steps, then bisection to well below a second. Unix time (UTC)
// or null when the body does not rise (set) in that time.
function findAltitudeCrossing(altFn, startUnix, endUnix, isRise) {
  const step = 300;
  let tPrev = startUnix, hPrev = altFn(startUnix) - RISE_SET_ALTITUDE;
  for (let t = startUnix + step; t <= endUnix + step - 1; t += step) {
    const tt = Math.min(t, endUnix);
    const h = altFn(tt) - RISE_SET_ALTITUDE;
    if (isRise ? (hPrev < 0 && h >= 0) : (hPrev > 0 && h <= 0)) {
      let left = tPrev, right = tt;
      for (let i = 0; i < 20; i++) {
        const mid = (left + right) / 2;
        const above = altFn(mid) - RISE_SET_ALTITUDE >= 0;
        if (above === isRise) right = mid; else left = mid;
      }
      return (left + right) / 2;
    }
    tPrev = tt; hPrev = h;
    if (tt >= endUnix) break;
  }
  return null;
}

// Rise and set on the calendar day of unixTime in the display time zone
// (23 or 25 hours long on the days of the DST change). Unix times (UTC),
// null when there is none that day. The offset argument is ignored.
function riseAndSetOfDay(body, unixTime, latDeg, lonDeg) {
  const dayStart = startOfDayInZone(unixTime);
  const dayEnd = startOfDayInZone(dayStart + 30 * 3600);
  const altFn = bodyAltitudeFn(body, latDeg, lonDeg);
  return { rise: findAltitudeCrossing(altFn, dayStart, dayEnd, true),
           set: findAltitudeCrossing(altFn, dayStart, dayEnd, false) };
}

function getMoonRiseAndSet(unixTime, latDeg, lonDeg, _timezoneOffsetSeconds = 0) {
  const r = riseAndSetOfDay('moon', unixTime, latDeg, lonDeg);
  return { moonRise: r.rise, moonSet: r.set };
}

function getSunRiseAndSet(unixTime, latDeg, lonDeg, _timezoneOffsetSeconds = 0) {
  const r = riseAndSetOfDay('sun', unixTime, latDeg, lonDeg);
  return { sunRise: r.rise, sunSet: r.set };
}

// Time from unixTime to the next rise and the next set (searched up to
// 48 hours ahead: the Moon rises about 50 minutes later every day), as
// "HH:MM", null if there is none in that time
function getTimeUntilRiseSet(body, unixTime, latDeg, lonDeg) {
  const altFn = bodyAltitudeFn(body, latDeg, lonDeg);
  const end = unixTime + 48 * 3600;
  const rise = findAltitudeCrossing(altFn, unixTime, end, true);
  const set = findAltitudeCrossing(altFn, unixTime, end, false);
  return {
    timeUntilRise: rise !== null ? formatTimeDuration(rise - unixTime) : null,
    timeUntilSet: set !== null ? formatTimeDuration(set - unixTime) : null
  };
}

// HH:MM in the display time zone (the offset argument is ignored)
function formatTimeString(unixTime, _timezoneOffsetSeconds = 0) {
  return formatClock(unixTime);
}

function formatTimeDuration(seconds) {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}`;
}

// Hamburger Menu Funktion
function toggleHamburgerMenu() {
  const navMenu = document.getElementById('nav-menu');
  const hamburgerBtn = document.getElementById('hamburger-menu-btn');

  navMenu.classList.toggle('nav-menu-open');
  hamburgerBtn.classList.toggle('hamburger-menu-open');
}

// Schließe Menü beim Klicken außerhalb
document.addEventListener('click', function (event) {
  const navMenu = document.getElementById('nav-menu');
  const hamburgerBtn = document.getElementById('hamburger-menu-btn');

  if (!navMenu.contains(event.target) && !hamburgerBtn.contains(event.target)) {
    navMenu.classList.remove('nav-menu-open');
    hamburgerBtn.classList.remove('hamburger-menu-open');
  }
});

// Einfache Luminanz-Histogramm-Implementierung (nur Helligkeit, nicht RGB)
function updateRGBHistogram(img, canvas) {
  if (!img || !img.complete || img.naturalWidth === 0 || !canvas) {
    return false;
  }
  
  try {
    // Feste Buffer-Größe; CSS width:100% skaliert den Canvas auf den Container.
    // clientWidth wird bewusst NICHT gemessen – das verursacht Timing-Probleme
    // wenn das Layout beim ersten Render-Frame noch nicht berechnet ist.
    const displayWidth  = 512;
    const displayHeight = 100;

    if (canvas.width  !== displayWidth)  canvas.width  = displayWidth;
    if (canvas.height !== displayHeight) canvas.height = displayHeight;

    const ctx = canvas.getContext('2d');
    const bins = 256;
    const histogram = new Array(bins).fill(0);

    // Kleines temporäres Canvas für schnelle Verarbeitung
    const tempCanvas = document.createElement('canvas');
    const scale = Math.min(1, 150 / Math.max(img.naturalWidth, img.naturalHeight));
    tempCanvas.width = Math.max(1, Math.round(img.naturalWidth * scale));
    tempCanvas.height = Math.max(1, Math.round(img.naturalHeight * scale));
    
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(img, 0, 0, tempCanvas.width, tempCanvas.height);
    const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
    const data = imageData.data;

    // Nur jeden 4. Pixel verarbeiten für Performance
    for (let i = 0; i < data.length; i += 16) { // 16 = 4 (RGBA) * 4 (skip)
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      
      // Pixel mit RGB=0 ignorieren
      if (r === 0 && g === 0 && b === 0) continue;
      
      // Luminanz berechnen (sRGB Formel)
      const luminance = Math.round(0.2126 * r + 0.7152 * g + 0.0722 * b);
      histogram[luminance]++;
    }

    // Maximum finden
    let maxCount = 0;
    for (let i = 0; i < bins; i++) {
      if (histogram[i] > maxCount) maxCount = histogram[i];
    }

    // Canvas leeren
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (maxCount === 0) {
      return false;
    }

    // Histogramm zeichnen (einfach, nur Luminanz)
    const barWidth = canvas.width / bins;
    ctx.fillStyle = '#4af'; // Blau
    
    for (let i = 0; i < bins; i++) {
      const height = (histogram[i] / maxCount) * canvas.height;
      ctx.fillRect(i * barWidth, canvas.height - height, barWidth, height);
    }

    // Beschriftungen 0 und 255 hinzufügen
    ctx.fillStyle = '#aaa';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('0', 4, 4);
    ctx.textAlign = 'right';
    ctx.fillText('255', canvas.width - 4, 4);

    // Statistiken berechnen
    let min = 255, max = 0;
    let sum = 0, pixelCount = 0;
    
    for (let i = 0; i < bins; i++) {
      if (histogram[i] > 0) {
        if (i < min) min = i;
        if (i > max) max = i;
        sum += i * histogram[i];
        pixelCount += histogram[i];
      }
    }

    let average = pixelCount > 0 ? sum / pixelCount : 0;
    
    // Median berechnen (effizienter: direkt aus Histogramm)
    let median = 0;
    if (pixelCount > 0) {
      const target = pixelCount / 2;
      let count = 0;
      let medianLow = 0, medianHigh = 0;
      
      for (let i = 0; i < bins; i++) {
        count += histogram[i];
        if (count >= target && medianLow === 0) {
          medianLow = i;
        }
        if (count >= target + 1) {
          medianHigh = i;
          break;
        }
      }
      
      median = pixelCount % 2 === 0 
        ? (medianLow + medianHigh) / 2 
        : medianLow;
    }

    // Statistiken im histogram-stats div anzeigen
    const statsEl = canvas.parentElement?.querySelector('#histogram-stats');
    if (statsEl) {
      statsEl.innerHTML = `
        <div style="display: flex; margin-bottom: 0.2rem;">
          <span style="min-width: 80px;">min:</span>
          <span>${min}</span>
        </div>
        <div style="display: flex; margin-bottom: 0.2rem;">
          <span style="min-width: 80px;">max:</span>
          <span>${max}</span>
        </div>
        <div style="display: flex; margin-bottom: 0.2rem;">
          <span style="min-width: 80px;">Average:</span>
          <span>${average.toFixed(1)}</span>
        </div>
        <div style="display: flex;">
          <span style="min-width: 80px;">Median:</span>
          <span>${median.toFixed(1)}</span>
        </div>
      `;
    }

    return true;
  } catch (error) {
    console.error("updateRGBHistogram error:", error);
    return false;
  }
}

// Wrapper function for processing.html compatibility
function updateHistogramForImage(imageId) {
  const img = document.getElementById(imageId);
  if (!img) return;

  const stepId = imageId.split('-')[1];
  const histCanvas = document.getElementById(`histogram-${stepId}`);
  if (!histCanvas) return;

  updateRGBHistogram(img, histCanvas);
}

// Zentrale Config-Ladefunktion mit Cache und In-Flight-Guard
let __configLoadingPromise = null;
let __configCache = null;

async function fetchConfig(force = false) {
  try {
    if (!force && __configCache) return __configCache;
    if (!force && __configLoadingPromise) return await __configLoadingPromise;

    __configLoadingPromise = (async () => {
      const resp = await fetch('api/config', { cache: 'no-cache' });
      if (!resp.ok) throw new Error(`config fetch failed: ${resp.status}`);
      const data = await resp.json();
      __configCache = data;
      setStationTimeZone(data.timezone || data.location_timezone);
      return data;
    })();

    return await __configLoadingPromise;
  } catch (err) {
    console.warn('config not found or failed to load', err);
    return null;
  } finally {
    __configLoadingPromise = null;
  }
}


// Learn the station time zone on every page (cached by fetchConfig)
if (typeof window !== 'undefined' && typeof fetch === 'function')
  fetchConfig();
