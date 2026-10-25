let loading;
export const loadECharts = () => loading ??= window.echarts ? Promise.resolve(window.echarts) : new Promise((ok, no) => {
  const s = document.createElement("script");
  s.src = new URL("../vendor/echarts/echarts.min.js", import.meta.url).href;
  s.onload = () => ok(window.echarts);
  s.onerror = () => no(new Error("Could not load ECharts"));
  document.head.append(s);
});
