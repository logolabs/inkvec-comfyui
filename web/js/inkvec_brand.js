/**
 * Inkvec branding: the Inkvec mark (the ink drop and its dot) above the INKVEC wordmark,
 * drawn in a reserved strip at the bottom of every Inkvec node. Copper on the node's own
 * ground, so it reads on light and dark ComfyUI themes alike. Purely visual -- remove
 * this file (and WEB_DIRECTORY in __init__.py) and the nodes work exactly as before.
 */
import { app } from "../../scripts/app.js";

const BRAND = {
  mark: "data:image/svg+xml;charset=utf-8,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20viewBox%3D%22-.5%20-.5%201024%201024%22%20width%3D%221024%22%20height%3D%221024%22%3E%3Cpath%20d%3D%22M511.3%20140.6c-23.3%2079.8-103%20178.5-144.9%20232.75L335.45%20415.4a467.73%20467.73%200%2000-64.8%20115.4c-65.5%20193.1%2080.8%20351.3%20237.85%20351.7%2021.26%201.1%2042.56-1.3%2063.06-7%2090.8-18.87%20211.87-120.04%20195.47-280.9-17.9-164.8-187.55-272.65-251.8-443.84Zm10.2%20374.9a61.05%2061.05%200%2000-67.1%2081.6%2061.05%2061.05%200%2000104.3%2017.26%2061.05%2061.05%200%2000-37.2-98.86ZM511.4%20817.65l177.16-201.3-43-.15a44.56%2044.56%200%2000-36.2%2018.47l-97.8%20111.67-102.2-116.67c-16.06-18.94-38.06-12-74.9-13.4%205.7%207.3%20159.75%20179.73%20176.93%20201.4Z%22%20fill%3D%22%23c9754a%22%20fill-rule%3D%22evenodd%22%2F%3E%3C%2Fsvg%3E",
  wordmark: "INKVEC",
  color: "#c9754a",
  strip: 56, // reserved at the bottom of the node, mark + gap + tracked caps
};

const mark = new Image();
let markReady = false;
mark.onload = () => {
  markReady = true;
};
mark.src = BRAND.mark;

app.registerExtension({
  name: "inkvec.brand",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!nodeData.name || !String(nodeData.name).startsWith("Inkvec")) return;

    // Reserve the strip in the size the frontend computes from the widgets, so the
    // brand never sits on top of one.
    const originalComputeSize = nodeType.prototype.computeSize;
    nodeType.prototype.computeSize = function (out) {
      const size = originalComputeSize
        ? originalComputeSize.apply(this, arguments)
        : out || [0, 0];
      return [size[0], size[1] + BRAND.strip];
    };

    nodeType.prototype.onDrawForeground = function (ctx) {
      if (!ctx || (this.flags && this.flags.collapsed)) return;
      const w = this.size && this.size[0];
      const h = this.size && this.size[1];
      if (!w || !h) return;
      const cx = w / 2;
      const markH = 34;
      const markW = markH * (514.25 / 742.25); // the mark's own viewBox aspect
      const top = h - BRAND.strip + 4;
      ctx.save();
      if (markReady) {
        ctx.drawImage(mark, cx - markW / 2, top, markW, markH);
      }
      ctx.fillStyle = BRAND.color;
      ctx.textAlign = "center";
      ctx.textBaseline = "alphabetic";
      try {
        ctx.letterSpacing = "3px";
      } catch (e) {
        /* older canvases: the wordmark just reads tighter */
      }
      ctx.font = "600 10px ui-monospace, Consolas, Menlo, monospace";
      ctx.fillText(BRAND.wordmark, cx + 1.5, h - 7);
      try {
        ctx.letterSpacing = "0px";
      } catch (e) {
        /* as above */
      }
      ctx.restore();
    };
  },
});
