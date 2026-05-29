function sortTable(tableId, colIndex) {
  const table = document.getElementById(tableId);
  if (!table) return;
  const thead = table.tHead;
  const th = thead ? thead.rows[0].cells[colIndex] : null;
  const type = (th && th.dataset.type) ? th.dataset.type : 'string';

  // toggle direction on header (store on element)
  const prevDir = th && th.dataset.sortDir;
  const dir = (prevDir === 'asc') ? 'desc' : 'asc';
  if (th) th.dataset.sortDir = dir;

  const tbody = table.tBodies[0];
  const rows = Array.from(tbody.querySelectorAll('tr'));

  const getCellText = (row) => {
    const cell = row.querySelectorAll('td')[colIndex];
    if (!cell) return '';

    // Prefer explicit sortable value if present
    const sortValue = cell.getAttribute('data-sort');
    if (sortValue !== null && sortValue !== undefined && sortValue !== '') {
      return sortValue.trim();
    }

    return cell.textContent.trim();
  };

  const parseDateValue = (text) => {
    const trimmed = text.trim();
    if (!trimmed) return 0;

    // Handle YYYY-MM-DD or YYYY/MM/DD
    const ymdMatch = trimmed.match(/^\s*(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?:[ T].*)?$/);
    if (ymdMatch) {
      const year = Number(ymdMatch[1]);
      const month = Number(ymdMatch[2]);
      const day = Number(ymdMatch[3]);
      if (month >= 1 && month <= 12 && day >= 1 && day <= 31) {
        return Date.UTC(year, month - 1, day);
      }
    }

    // Handle DD/MM/YYYY or D/M/YYYY explicitly
    const dmyMatch = trimmed.match(/^\s*(\d{1,2})\/(\d{1,2})\/(\d{4})\s*$/);
    if (dmyMatch) {
      const day = Number(dmyMatch[1]);
      const month = Number(dmyMatch[2]);
      const year = Number(dmyMatch[3]);
      if (month >= 1 && month <= 12 && day >= 1 && day <= 31) {
        return Date.UTC(year, month - 1, day);
      }
    }

    // Handle DD.MM.YYYY
    const dmyDotMatch = trimmed.match(/^\s*(\d{1,2})\.(\d{1,2})\.(\d{4})\s*$/);
    if (dmyDotMatch) {
      const day = Number(dmyDotMatch[1]);
      const month = Number(dmyDotMatch[2]);
      const year = Number(dmyDotMatch[3]);
      if (month >= 1 && month <= 12 && day >= 1 && day <= 31) {
        return Date.UTC(year, month - 1, day);
      }
    }

    const t = Date.parse(trimmed);
    return isNaN(t) ? 0 : t;
  };

  const parseValue = (text) => {
    if (type === 'number') {
      // remove common extraneous chars (commas, percent signs, currency symbols)
      const n = parseFloat(text.replace(/[,€$£\s%]/g, '').replace(/[^0-9.\-eE+]/g, ''));
      return isNaN(n) ? -Infinity : n;
    }
    if (type === 'date') {
      return parseDateValue(text);
    }
    // default: case-insensitive string
    return text.toLowerCase();
  };

  rows.sort((a, b) => {
    const va = parseValue(getCellText(a));
    const vb = parseValue(getCellText(b));

    if (va === vb) return 0;

    // numeric/date comparisons will be numbers; strings are compared lexicographically
    if (typeof va === 'number' && typeof vb === 'number') {
      return dir === 'asc' ? va - vb : vb - va;
    }

    return dir === 'asc' ? (va > vb ? 1 : -1) : (va < vb ? 1 : -1);
  });

  // append rows in new order
  rows.forEach(r => tbody.appendChild(r));

  // update aria-sort on headers for accessibility & simple visual feedback
  if (thead) {
    Array.from(thead.rows[0].cells).forEach((cell, idx) => {
      cell.removeAttribute('aria-sort');
      if (idx === colIndex) {
        cell.setAttribute('aria-sort', dir === 'asc' ? 'ascending' : 'descending');
      }
    });
  }
}
