/*****************************************************************************
 *
 * Copyright (c) 2025 Sven Kreiensen
 * All rights reserved.
 *
 * You can use this software under the terms of the MIT license
 * (see LICENSE.md).
 *
 *****************************************************************************/
/*
 * System health for status.html and the status dot in the navigation
 * (GET /api/system/status): services and timers with their last run, age of
 * the latest image and of the fetched data, temperatures, load, memory, disk
 * space, database size, and a list of checks (ok / warn / error / off).
 *
 * Everything is read without privileges: "systemctl show" and
 * "systemctl list-timers" over D-Bus, /sys/class/thermal, statfs, os.
 */
'use strict';

const fs = require('fs');
const fsp = fs.promises;
const os = require('os');
const path = require('path');

// Expected run interval of the timer-driven services (OnUnitActiveSec in
// etc/systemd/system/*.timer); a last run older than STALE_FACTOR times
// this is reported
const TIMER_INTERVAL_S = {
    'allsky360-meteo.service': 30 * 60,
    'allsky360-aurora.service': 5 * 60,
    'allsky360-mount-position.service': 60,
    'allsky360-sensors.service': 5 * 60
};
const STALE_FACTOR = 3;

const LABELS = {
    'allsky360-camera.service': 'Camera',
    'allsky360-api.service': 'Web API',
    'allsky360-meteo.service': 'Meteo (Open-Meteo)',
    'allsky360-aurora.service': 'Aurora (NOAA/ESA)',
    'allsky360-mount-position.service': 'Mount position',
    'allsky360-sensors.service': 'Station sensors'
};

const IMAGE_WARN_S = 10 * 60;      // latest image older: warning
const IMAGE_ERROR_S = 30 * 60;     // ... error
const DISK_WARN_PCT = 10;          // free space below: warning
const DISK_ERROR_PCT = 5;          // ... error
const TEMP_WARN_C = 85;            // SoC junction temperature: warning
const TEMP_ERROR_C = 95;           // ... error (the Orin throttles near 99 °C)
const CABINET_DEW_MARGIN_K = 2;    // cabinet air closer to its dew point: warning

const CACHE_MS = 5000;             // pages poll every minute, several tabs share one answer

/** Parse "systemctl show" output of several units (blocks separated by blank lines). */
function parseShow(stdout) {
    const units = {};
    for (const block of stdout.split(/\n\s*\n/)) {
        const props = {};
        for (const line of block.split('\n')) {
            const i = line.indexOf('=');
            if (i > 0) props[line.slice(0, i)] = line.slice(i + 1);
        }
        if (props.Id) units[props.Id] = props;
    }
    return units;
}

/** "@1793717153" (--timestamp=unix) -> 1793717153, empty -> null */
function unixProp(v) {
    const m = /^@(\d+)/.exec(v || '');
    return m ? Number(m[1]) : null;
}

async function readJson(file) {
    try { return JSON.parse(await fsp.readFile(file, 'utf8')); } catch { return null; }
}

async function readTemperatures() {
    const base = '/sys/class/thermal';
    let zones = [];
    try { zones = (await fsp.readdir(base)).filter(z => z.startsWith('thermal_zone')); } catch { return []; }
    const out = [];
    for (const z of zones) {
        try {
            const type = (await fsp.readFile(path.join(base, z, 'type'), 'utf8')).trim();
            const raw = (await fsp.readFile(path.join(base, z, 'temp'), 'utf8')).trim();
            const milli = Number(raw);
            if (raw !== '' && Number.isFinite(milli))
                out.push({ name: type.replace(/-thermal$/, ''), celsius: Math.round(milli / 100) / 10 });
        } catch { /* zones of switched-off engines have no reading */ }
    }
    return out;
}

/** Free space of the file systems holding the given directories (each once). */
async function readDisks(dirs) {
    const seen = new Map();
    for (const [name, dir] of Object.entries(dirs)) {
        try {
            const st = await fsp.stat(dir);
            const s = await fsp.statfs(dir);
            const total = s.blocks * s.bsize, free = s.bavail * s.bsize;
            const key = String(st.dev);
            if (seen.has(key)) { seen.get(key).used_by.push(name); continue; }
            seen.set(key, { used_by: [name], total, free, free_pct: total ? Math.round(free / total * 1000) / 10 : null });
        } catch { /* directory missing: not installed yet */ }
    }
    return [...seen.values()];
}

async function readDatabases(dbDir) {
    try {
        const files = (await fsp.readdir(dbDir)).filter(f => /^database_\d{8}\.db$/.test(f)).sort();
        let bytes = 0;
        for (const f of files) {
            try { bytes += (await fsp.stat(path.join(dbDir, f))).size; } catch { /* removed meanwhile */ }
        }
        return { count: files.length, bytes, oldest: files[0]?.slice(9, 17) || null, newest: files.at(-1)?.slice(9, 17) || null };
    } catch {
        return null;
    }
}

function ageText(s) {
    if (s < 90) return `${Math.round(s)} s`;
    if (s < 5400) return `${Math.round(s / 60)} min`;
    if (s < 172800) return `${Math.round(s / 3600)} h`;
    return `${Math.round(s / 86400)} d`;
}

/**
 * createSystemStatus({ systemctl, execFileAsync, services, timerUnits,
 *                      processPatterns, publicPath, dbPath })
 * returns an async function producing the status object (cached CACHE_MS).
 */
function createSystemStatus(opts) {
    const { systemctl, execFileAsync, services, timerUnits, processPatterns, publicPath, dbPath } = opts;
    let cache = null, cacheTime = 0, pending = null;

    async function readServices() {
        const units = services.concat(Object.values(timerUnits));
        const props = 'Id,Description,ActiveState,SubState,Result,UnitFileState,' +
                      'ExecMainStartTimestamp,ExecMainExitTimestamp,ExecMainStatus';
        let show = {}, timers = [];
        try {
            const { stdout } = await execFileAsync(systemctl,
                ['--no-ask-password', '--timestamp=unix', 'show', '-p', props, ...units], { timeout: 10000 });
            show = parseShow(stdout);
        } catch { /* reported per unit as "unknown" */ }
        try {
            const { stdout } = await execFileAsync(systemctl,
                ['--no-ask-password', 'list-timers', '--all', '--output=json', ...Object.values(timerUnits)], { timeout: 10000 });
            timers = JSON.parse(stdout);
        } catch { /* older systemd without JSON output: no next/last */ }

        const result = [];
        for (const unit of services) {
            const p = show[unit] || {};
            const svc = {
                unit,
                label: LABELS[unit] || unit,
                description: p.Description || '',
                active: p.ActiveState || 'unknown',
                sub: p.SubState || '',
                result: p.Result || '',
                enabled: p.UnitFileState || '',
                last_start: unixProp(p.ExecMainStartTimestamp),
                last_exit: unixProp(p.ExecMainExitTimestamp),
                exit_status: p.ExecMainStatus !== undefined && p.ExecMainStatus !== '' ? Number(p.ExecMainStatus) : null
            };
            const timerUnit = timerUnits[unit];
            if (timerUnit) {
                const tp = show[timerUnit] || {};
                const lt = timers.find(t => t.unit === timerUnit) || {};
                svc.timer = {
                    unit: timerUnit,
                    active: tp.ActiveState || 'unknown',
                    enabled: tp.UnitFileState || '',
                    // list-timers: microseconds, 0/null when there is none
                    next: lt.next ? Math.round(lt.next / 1e6) : null,
                    last: lt.last ? Math.round(lt.last / 1e6) : null,
                    interval: TIMER_INTERVAL_S[unit] || null
                };
            }
            if (processPatterns[unit]) {
                try {
                    await execFileAsync('pgrep', ['-f', processPatterns[unit]], { timeout: 5000 });
                    svc.process = true;
                } catch {
                    svc.process = false;
                }
            }
            result.push(svc);
        }
        return result;
    }

    async function build() {
        const now = Date.now() / 1000;
        const [svcs, temps, disks, dbs, latestImage, sensors, meteo, aurora, forecast] = await Promise.all([
            readServices(),
            readTemperatures(),
            readDisks({ images: path.join(publicPath, 'images'), database: dbPath, system: '/' }),
            readDatabases(dbPath),
            readJson(path.join(publicPath, 'images', 'latest_image.json')),
            readJson(path.join(publicPath, 'latest_sensor_data.json')),
            readJson(path.join(publicPath, 'latest_meteo_data.json')),
            readJson(path.join(publicPath, 'latest_aurora_data.json')),
            readJson(path.join(publicPath, 'forecast_meteo.json'))
        ]);

        const checks = [];
        const add = (id, label, level, text) => checks.push({ id, label, level, text });

        // Services and timers
        for (const s of svcs) {
            if (s.timer) {
                const t = s.timer;
                if (t.active !== 'active') {
                    // a timer that was never enabled is a choice (e.g. no mount), not a fault
                    if (t.enabled === 'enabled') add(s.unit, s.label, 'error', `timer is ${t.active} although enabled`);
                    else add(s.unit, s.label, 'off', 'timer switched off');
                    continue;
                }
                if (!t.next && s.active !== 'activating') {
                    add(s.unit, s.label, 'error', 'timer active but no next run scheduled (restart the timer)');
                    continue;
                }
                if (s.active === 'failed' || (s.result && s.result !== 'success')) {
                    add(s.unit, s.label, 'error', `last run failed (${s.result || s.active}${s.exit_status ? ', exit ' + s.exit_status : ''})`);
                    continue;
                }
                const last = t.last || s.last_start;
                if (t.interval && last && now - last > STALE_FACTOR * t.interval) {
                    add(s.unit, s.label, 'warn', `last run ${ageText(now - last)} ago`);
                    continue;
                }
                add(s.unit, s.label, 'ok', last ? `last run ${ageText(now - last)} ago` : 'scheduled');
            } else if (s.active === 'active' || s.active === 'activating' || s.active === 'reloading') {
                add(s.unit, s.label, 'ok', 'running');
            } else if (s.process) {
                add(s.unit, s.label, 'warn', 'running outside systemd (make camera-run)');
            } else if (s.active === 'failed') {
                add(s.unit, s.label, 'error', `failed (${s.result})`);
            } else {
                add(s.unit, s.label, s.enabled === 'enabled' ? 'error' : 'warn', `not running (${s.active})`);
            }
        }

        // Latest image
        let image = null;
        if (latestImage && Number.isFinite(latestImage.image_timestamp)) {
            const age = now - latestImage.image_timestamp;
            image = { timestamp: latestImage.image_timestamp, age_s: Math.round(age),
                      path: `images/${latestImage.date}/image-${latestImage.date}${latestImage.image}.${latestImage.filetype || 'jpg'}` };
            add('image', 'Latest image', age > IMAGE_ERROR_S ? 'error' : age > IMAGE_WARN_S ? 'warn' : 'ok', `${ageText(age)} old`);
        } else {
            add('image', 'Latest image', 'error', 'no latest_image.json');
        }

        // Age of the data files of the fetch tools (the pages show these)
        // (time field of the file, else its modification time:
        // latest_meteo_data.json has no time field)
        const data = {};
        const dataAge = async (key, obj, field, file) => {
            let ts = obj && Number.isFinite(obj[field]) ? obj[field] : null;
            if (ts === null && obj) {
                try { ts = Math.round((await fsp.stat(path.join(publicPath, file))).mtimeMs / 1000); } catch { /* gone */ }
            }
            data[key] = ts === null ? null : { timestamp: ts, age_s: Math.round(now - ts) };
        };
        await Promise.all([
            dataAge('sensors', sensors, 'timestamp', 'latest_sensor_data.json'),
            dataAge('meteo', meteo, 'timestamp', 'latest_meteo_data.json'),
            dataAge('aurora', aurora, 'observation_time', 'latest_aurora_data.json'),
            dataAge('forecast', forecast, 'generated', 'forecast_meteo.json')
        ]);

        // Temperatures: the junction (tj) is the hottest point of the Orin
        const tj = temps.find(t => t.name === 'tj') ||
                   temps.reduce((m, t) => (!m || t.celsius > m.celsius ? t : m), null);
        if (tj) add('temperature', 'SoC temperature', tj.celsius >= TEMP_ERROR_C ? 'error' : tj.celsius >= TEMP_WARN_C ? 'warn' : 'ok', `${tj.celsius.toFixed(1)} °C`);

        // Disk space
        for (const d of disks) {
            if (d.free_pct === null) continue;
            add(`disk-${d.used_by.join('-')}`, `Disk (${d.used_by.join(', ')})`,
                d.free_pct < DISK_ERROR_PCT ? 'error' : d.free_pct < DISK_WARN_PCT ? 'warn' : 'ok',
                `${d.free_pct.toFixed(1)} % free`);
        }

        // Control cabinet climate (station sensors, only when present)
        let cabinet = null;
        if (sensors && Number.isFinite(sensors.cabinet_temperature)) {
            cabinet = {
                temperature: sensors.cabinet_temperature,
                humidity: Number.isFinite(sensors.cabinet_humidity) ? sensors.cabinet_humidity : null,
                dew_point: Number.isFinite(sensors.cabinet_dew_point) ? sensors.cabinet_dew_point : null,
                timestamp: sensors.timestamp
            };
            if (cabinet.dew_point !== null) {
                const margin = cabinet.temperature - cabinet.dew_point;
                add('cabinet', 'Control cabinet', margin < CABINET_DEW_MARGIN_K ? 'warn' : 'ok',
                    `${margin.toFixed(1)} K above dew point${margin < CABINET_DEW_MARGIN_K ? ' · condensation risk' : ''}`);
            }
        }

        const levels = checks.map(c => c.level);
        const overall = levels.includes('error') ? 'error' : levels.includes('warn') ? 'warn' : 'ok';

        return {
            generated: Math.round(now),
            overall,
            checks,
            services: svcs,
            image,
            data,
            host: {
                hostname: os.hostname(),
                uptime_s: Math.round(os.uptime()),
                load: os.loadavg().map(v => Math.round(v * 100) / 100),
                cpus: os.cpus().length,
                mem_total: os.totalmem(),
                mem_available: os.freemem()
            },
            temperatures: temps,
            disks,
            databases: dbs,
            cabinet
        };
    }

    return async function systemStatus() {
        if (cache && Date.now() - cacheTime < CACHE_MS) return cache;
        if (!pending) {
            pending = build().then(r => { cache = r; cacheTime = Date.now(); return r; })
                             .finally(() => { pending = null; });
        }
        return pending;
    };
}

module.exports = { createSystemStatus, parseShow };
