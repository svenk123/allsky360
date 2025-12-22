// POI-Liste für sichtbare Kometen
// Format: Array von Objekten mit name, ra, dec und optional color
let cometsPoi = [];

/**
 * Ruft die aktuell sichtbaren Kometen von der COBS API ab
 * @param {number} longitude - Längengrad des Beobachters
 * @param {number} latitude - Breitengrad des Beobachters
 * @param {number} altitude - Höhe des Beobachters in Metern
 * @param {string|Date} date - Datum im Format 'YYYY-MM-DD' oder 'YYYY-MM-DD hh:mm' oder Date-Objekt
 * @returns {Promise<Array>} Promise, das mit der aktualisierten cometsPoi Liste aufgelöst wird
 */
async function fetchComets(longitude, latitude, altitude, date) {
  try {
    // Datum formatieren
    let dateString;
    if (date instanceof Date) {
      const year = date.getUTCFullYear();
      const month = String(date.getUTCMonth() + 1).padStart(2, '0');
      const day = String(date.getUTCDate()).padStart(2, '0');
      dateString = `${year}-${month}-${day}`;
    } else {
      dateString = date;
    }

    // API-URL zusammenstellen (lokaler Proxy-Endpoint)
    const apiUrl = `/api/comets?date=${encodeURIComponent(dateString)}&lat=${latitude}&long=${longitude}&alt=${altitude}`;

    // API-Aufruf über lokalen Proxy (umgeht CORS-Probleme)
    const response = await fetch(apiUrl);

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();

    // Prüfen ob die API einen Fehler zurückgegeben hat
    if (data.code && data.code !== "200") {
      console.warn('COBS API Fehler:', data.message || 'Unbekannter Fehler');
      cometsPoi = [];
      return cometsPoi;
    }

    // Kometen-Daten in POI-Format konvertieren
    cometsPoi = [];
    
    // Die API gibt ein Array von Kometen zurück
    // Parse comet_fullname, best_ra und best_dec
    if (Array.isArray(data)) {
      data.forEach(comet => {
        if (comet.comet_fullname && comet.best_ra !== undefined && comet.best_dec !== undefined) {
          cometsPoi.push({
            name: comet.comet_fullname,
            ra: parseFloat(comet.best_ra),
            dec: parseFloat(comet.best_dec),
            color: "cyan" // Standardfarbe für Kometen
          });
        }
      });
    } else if (data.comets && Array.isArray(data.comets)) {
      // Falls die API die Kometen in einem 'comets' Objekt zurückgibt
      data.comets.forEach(comet => {
        if (comet.comet_fullname && comet.best_ra !== undefined && comet.best_dec !== undefined) {
          cometsPoi.push({
            name: comet.comet_fullname,
            ra: parseFloat(comet.best_ra),
            dec: parseFloat(comet.best_dec),
            color: "cyan"
          });
        }
      });
    }

    return cometsPoi;
  } catch (error) {
    console.error('Fehler beim Abrufen der Kometen:', error);
    cometsPoi = [];
    return cometsPoi;
  }
}

