const byId = id => document.getElementById(id);
let token = '';
let submitted = false;
let stopRequested = false;

async function refresh() {
  try {
    const response = await fetch('/api/status', {cache: 'no-store'});
    const state = await response.json();
    token = state.token;
    byId('connection').classList.toggle('online', state.cameras_online);
    byId('connection').lastElementChild.textContent = state.cameras_online ? '双路相机在线' : '等待相机连接';
    byId('run').disabled = state.active || submitted || !state.cameras_online;
    byId('resume').hidden = !state.has_checkpoint || state.active;
    byId('resume').disabled = state.active || submitted || !state.cameras_online;
    byId('stop').disabled = !state.active || stopRequested;
    if (!state.active) stopRequested = false;
    const total = state.total_steps || 3;
    byId('step').textContent = `${state.step || 0} / ${total}`;
    const checked = state.events.filter(entry => entry.state === 'checked').length;
    byId('progress').max = total;
    byId('progress').value = state.status === 'complete' ? total : Math.max(checked, (state.step || 0) - (state.status === 'checked' ? 0 : 1));
    byId('message').textContent = state.message;
    const lines = state.events.slice(-12).reverse().map(entry => {
      const line = document.createElement('li');
      const stamp = document.createElement('time');
      const content = document.createElement('span');
      stamp.textContent = entry.time;
      content.textContent = entry.message;
      line.append(stamp, content);
      return line;
    });
    byId('events').replaceChildren(...lines);
  } catch {
    byId('connection').classList.remove('online');
    byId('connection').lastElementChild.textContent = '控制台连接中断';
    byId('run').disabled = true;
  }
  setTimeout(refresh, 750);
}

function view(name) {
  const incoming = new Image();
  const url = `/camera/${name}.jpg?t=${Date.now()}`;
  incoming.onload = () => { byId(name).src = url; setTimeout(() => view(name), 380); };
  incoming.onerror = () => { setTimeout(() => view(name), 1000); };
  incoming.src = url;
}

async function post(route, data) {
  const response = await fetch(route, {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Console-Token': token}, body: JSON.stringify(data)});
  const body = await response.json();
  if (!response.ok) throw new Error(body.error);
}

byId('command-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (submitted) return;
  submitted = true;
  byId('run').disabled = true;
  byId('feedback').textContent = '';
  try { await post('/api/run', {instruction: byId('instruction').value}); }
  catch (error) { byId('feedback').textContent = error.message; }
  finally { submitted = false; }
});

byId('resume').addEventListener('click', async () => {
  byId('instruction').value = '继续上次换位';
  byId('command-form').requestSubmit();
});

byId('stop').addEventListener('click', async () => {
  try {
    await post('/api/stop', {});
    stopRequested = true;
    byId('feedback').textContent = '已请求在当前动作安全结束后停止。';
  } catch (error) { byId('feedback').textContent = error.message; }
});

refresh();
view('fixed');
view('wrist');
