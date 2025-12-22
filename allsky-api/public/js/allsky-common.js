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

function formatTimestamp(unixTime, timezoneOffsetSeconds = null) {
  if (timezoneOffsetSeconds !== null) {
    // Use server timezone offset
    const date = new Date((unixTime + timezoneOffsetSeconds) * 1000);
    // UTC methods to format without browser timezone interference
    const year = date.getUTCFullYear();
    const month = String(date.getUTCMonth() + 1).padStart(2, '0');
    const day = String(date.getUTCDate()).padStart(2, '0');
    const hours = String(date.getUTCHours()).padStart(2, '0');
    const minutes = String(date.getUTCMinutes()).padStart(2, '0');
    const seconds = String(date.getUTCSeconds()).padStart(2, '0');
    return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
  } else {
    // Use browser's local timezone
    const date = new Date(unixTime * 1000);
    return date.toLocaleString();
  }
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
  return night ? "🌙" : "☀️";
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

  // Fallback (sollte nie erreicht werden)
  return "rgb(240,240,200)";
}

function getSeeingLabel(jetstream_kmh) {
  let label = "";
  let color = "";

  if (jetstream_kmh <= 30) {
    label = "excellent";
    color = "green";
  } else if (jetstream_kmh <= 60) {
    label = "good";
    color = "limegreen";
  } else if (jetstream_kmh <= 90) {
    label = "fair";
    color = "orange";
  } else if (jetstream_kmh <= 120) {
    label = "poor";
    color = "orangered";
  } else {
    label = "very poor";
    color = "red";
  }

  return `<span style="color: ${color}">(${label})</span>`;
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

function azAltToImageXY(azimuthDeg, altitudeDeg) {
  azimuthDeg = (360 - azimuthDeg) % 360;

  const scaleX = imageEl.clientWidth / imageEl.naturalWidth;
  const scaleY = imageEl.clientHeight / imageEl.naturalHeight;

  let r = crosshairRadius * scaleX;
  if (projectionType === "sinus") {
    r *= Math.sin((90 - altitudeDeg) * Math.PI / 180);
  } else {
    r *= (90 - altitudeDeg) / 90;
  }

  const angleRad = (azimuthDeg + crosshairRotation) * Math.PI / 180;
  const cx = (crosshairCenterX) * scaleX * zoomFactor + panX;
  const cy = (crosshairCenterY) * scaleY * zoomFactor + panY;

  const x = cx + Math.sin(angleRad) * r * zoomFactor;
  const y = cy - Math.cos(angleRad) * r * zoomFactor;

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
  if (altitude < 0) return; // unter Horizont → nicht zeichnen

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

  return { ra, dec: decRad * RAD2DEG };
}

// Calculate moon rise and set times
function getMoonRiseAndSet(unixTime, latDeg, lonDeg, timezoneOffsetSeconds = 0) {
  // Convert UTC unix time to local time
  const localTime = unixTime + timezoneOffsetSeconds;
  
  // Get start of day in local time
  const localDate = new Date(localTime * 1000);
  const startOfDay = new Date(localDate.getFullYear(), localDate.getMonth(), localDate.getDate());
  const startOfDayUnix = Math.floor(startOfDay.getTime() / 1000) - timezoneOffsetSeconds;
  
  // Search for moon rise and set during this day
  const moonRise = findMoonRiseOrSet(startOfDayUnix, latDeg, lonDeg, timezoneOffsetSeconds, true);
  const moonSet = findMoonRiseOrSet(startOfDayUnix, latDeg, lonDeg, timezoneOffsetSeconds, false);
  
  return {
    moonRise: moonRise,
    moonSet: moonSet
  };
}

function findMoonRiseOrSet(dayStartUnix, latDeg, lonDeg, timezoneOffsetSeconds, isRise) {
  const searchWindow = 24 * 3600; // 24 hours in seconds
  const stepSize = 300; // 5 minutes in seconds
  let lastAltitude = null;
  
  for (let t = dayStartUnix; t < dayStartUnix + searchWindow; t += stepSize) {
    const moon = getMoonRaDec(t);
    const { altitude } = raDecToAzAlt(moon.ra, moon.dec, latDeg, lonDeg, t);
    
    // Look for transition from negative to positive altitude (rise) or positive to negative (set)
    if (lastAltitude !== null) {
      if (isRise && lastAltitude < 0 && altitude >= 0) {
        // Found moon rise - refine with smaller steps
        return findExactTime(t - stepSize, t, latDeg, lonDeg, timezoneOffsetSeconds, true);
      } else if (!isRise && lastAltitude > 0 && altitude <= 0) {
        // Found moon set - refine with smaller steps
        return findExactTime(t - stepSize, t, latDeg, lonDeg, timezoneOffsetSeconds, false);
      }
    }
    
    lastAltitude = altitude;
  }
  
  return null; // No rise/set found (polar day/night or moon never rises/sets)
}

function findExactTime(startUnix, endUnix, latDeg, lonDeg, timezoneOffsetSeconds, isRise) {
  // Binary search for exact rise/set time
  let left = startUnix;
  let right = endUnix;
  
  for (let i = 0; i < 20; i++) { // 20 iterations should be enough
    const mid = (left + right) / 2;
    const moon = getMoonRaDec(mid);
    const { altitude } = raDecToAzAlt(moon.ra, moon.dec, latDeg, lonDeg, mid);
    
    if (isRise) {
      if (altitude >= 0) {
        right = mid;
      } else {
        left = mid;
      }
    } else {
      if (altitude <= 0) {
        right = mid;
      } else {
        left = mid;
      }
    }
  }
  
  const resultTime = (left + right) / 2 + timezoneOffsetSeconds;
  return resultTime;
}

// Format time as HH:MM in local time
function formatTimeString(unixTime, timezoneOffsetSeconds = 0) {
  const localTime = unixTime + timezoneOffsetSeconds;
  const date = new Date(localTime * 1000);
  const hours = String(date.getUTCHours()).padStart(2, '0');
  const minutes = String(date.getUTCMinutes()).padStart(2, '0');
  return `${hours}:${minutes}`;
}

// Calculate time until moon rise or set
function getTimeUntilMoonRiseSet(moonRiseUnix, moonSetUnix, currentUnix, timezoneOffsetSeconds = 0) {
  if (!moonRiseUnix && !moonSetUnix) {
    return { timeUntilRise: null, timeUntilSet: null };
  }
  
  const localCurrent = currentUnix + timezoneOffsetSeconds;
  let timeUntilRise = null;
  let timeUntilSet = null;
  
  if (moonRiseUnix) {
    const localRise = moonRiseUnix + timezoneOffsetSeconds;
    if (localRise >= localCurrent) {
      timeUntilRise = localRise - localCurrent;
    } else {
      // Moon already rose today, calculate time until next rise (next day)
      timeUntilRise = (localRise + 24 * 3600) - localCurrent;
    }
  }
  
  if (moonSetUnix) {
    const localSet = moonSetUnix + timezoneOffsetSeconds;
    if (localSet >= localCurrent) {
      timeUntilSet = localSet - localCurrent;
    } else {
      // Moon already set today, calculate time until next set (next day)
      timeUntilSet = (localSet + 24 * 3600) - localCurrent;
    }
  }
  
  return {
    timeUntilRise: timeUntilRise ? formatTimeDuration(timeUntilRise) : null,
    timeUntilSet: timeUntilSet ? formatTimeDuration(timeUntilSet) : null
  };
}

function formatTimeDuration(seconds) {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}`;
}

// Calculate sun rise and set times
function getSunRiseAndSet(unixTime, latDeg, lonDeg, timezoneOffsetSeconds = 0) {
  // Convert UTC unix time to local time
  const localTime = unixTime + timezoneOffsetSeconds;
  
  // Get start of day in local time
  const localDate = new Date(localTime * 1000);
  const startOfDay = new Date(localDate.getFullYear(), localDate.getMonth(), localDate.getDate());
  const startOfDayUnix = Math.floor(startOfDay.getTime() / 1000) - timezoneOffsetSeconds;
  
  // Search for sun rise and set during this day
  const sunRise = findSunRiseOrSet(startOfDayUnix, latDeg, lonDeg, timezoneOffsetSeconds, true);
  const sunSet = findSunRiseOrSet(startOfDayUnix, latDeg, lonDeg, timezoneOffsetSeconds, false);
  
  return {
    sunRise: sunRise,
    sunSet: sunSet
  };
}

function findSunRiseOrSet(dayStartUnix, latDeg, lonDeg, timezoneOffsetSeconds, isRise) {
  const searchWindow = 24 * 3600; // 24 hours in seconds
  const stepSize = 300; // 5 minutes in seconds
  let lastAltitude = null;
  
  for (let t = dayStartUnix; t < dayStartUnix + searchWindow; t += stepSize) {
    const sun = getSunRaDec(t);
    const { altitude } = raDecToAzAlt(sun.ra, sun.dec, latDeg, lonDeg, t);
    
    // Look for transition from negative to positive altitude (rise) or positive to negative (set)
    if (lastAltitude !== null) {
      if (isRise && lastAltitude < 0 && altitude >= 0) {
        // Found sun rise - refine with smaller steps
        return findExactSunTime(t - stepSize, t, latDeg, lonDeg, timezoneOffsetSeconds, true);
      } else if (!isRise && lastAltitude > 0 && altitude <= 0) {
        // Found sun set - refine with smaller steps
        return findExactSunTime(t - stepSize, t, latDeg, lonDeg, timezoneOffsetSeconds, false);
      }
    }
    
    lastAltitude = altitude;
  }
  
  return null; // No rise/set found (polar day/night or sun never rises/sets)
}

function findExactSunTime(startUnix, endUnix, latDeg, lonDeg, timezoneOffsetSeconds, isRise) {
  // Binary search for exact rise/set time
  let left = startUnix;
  let right = endUnix;
  
  for (let i = 0; i < 20; i++) { // 20 iterations should be enough
    const mid = (left + right) / 2;
    const sun = getSunRaDec(mid);
    const { altitude } = raDecToAzAlt(sun.ra, sun.dec, latDeg, lonDeg, mid);
    
    if (isRise) {
      if (altitude >= 0) {
        right = mid;
      } else {
        left = mid;
      }
    } else {
      if (altitude <= 0) {
        right = mid;
      } else {
        left = mid;
      }
    }
  }
  
  const resultTime = (left + right) / 2 + timezoneOffsetSeconds;
  return resultTime;
}

// Calculate time until sun rise or set
function getTimeUntilSunRiseSet(sunRiseUnix, sunSetUnix, currentUnix, timezoneOffsetSeconds = 0) {
  if (!sunRiseUnix && !sunSetUnix) {
    return { timeUntilRise: null, timeUntilSet: null };
  }
  
  const localCurrent = currentUnix + timezoneOffsetSeconds;
  let timeUntilRise = null;
  let timeUntilSet = null;
  
  if (sunRiseUnix) {
    const localRise = sunRiseUnix + timezoneOffsetSeconds;
    if (localRise >= localCurrent) {
      timeUntilRise = localRise - localCurrent;
    } else {
      // Sun already rose today, calculate time until next rise (next day)
      timeUntilRise = (localRise + 24 * 3600) - localCurrent;
    }
  }
  
  if (sunSetUnix) {
    const localSet = sunSetUnix + timezoneOffsetSeconds;
    if (localSet >= localCurrent) {
      timeUntilSet = localSet - localCurrent;
    } else {
      // Sun already set today, calculate time until next set (next day)
      timeUntilSet = (localSet + 24 * 3600) - localCurrent;
    }
  }
  
  return {
    timeUntilRise: timeUntilRise ? formatTimeDuration(timeUntilRise) : null,
    timeUntilSet: timeUntilSet ? formatTimeDuration(timeUntilSet) : null
  };
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
    // Canvas-Größe anpassen
    const container = canvas.parentElement;
    const displayWidth = container ? (container.clientWidth - 32) : 512;
    const displayHeight = 100;
    
    if (canvas.width !== displayWidth) canvas.width = displayWidth;
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
      return data;
    })();

    return await __configLoadingPromise;
  } catch (err) {
    console.warn('⚠️ config not found or failed to load', err);
    return null;
  } finally {
    __configLoadingPromise = null;
  }
}

