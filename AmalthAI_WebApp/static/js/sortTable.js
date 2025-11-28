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
    return cell ? cell.textContent.trim() : '';
  };

  const parseValue = (text) => {
    if (type === 'number') {
      // remove common extraneous chars (commas, percent signs, currency symbols)
      const n = parseFloat(text.replace(/[,€$£\s%]/g, '').replace(/[^0-9.\-eE+]/g, ''));
      return isNaN(n) ? -Infinity : n;
    }
    if (type === 'date') {
      const t = Date.parse(text);
      return isNaN(t) ? 0 : t;
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
