/**
 * Inkvec branding: the Inkvec mark (the ink drop and its dot) above the INKVEC wordmark,
 * drawn in a reserved strip at the bottom of every Inkvec node. Copper on the node's own
 * ground, so it reads on light and dark ComfyUI themes alike. Purely visual -- remove
 * this file (and WEB_DIRECTORY in __init__.py) and the nodes work exactly as before.
 */
import { app } from "../../scripts/app.js";

const BRAND = {
  mark: "data:image/svg+xml;charset=utf-8,%3Csvg%20width%3D%22427%22%20height%3D%22597%22%20viewBox%3D%220%200%20427%20597%22%20fill%3D%22none%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%3E%3Cpath%20fill-rule%3D%22evenodd%22%20clip-rule%3D%22evenodd%22%20d%3D%22M211.03%201.72%20208.21%209.5C153.7%20160.07-16.65%20252.01%2015.25%20420.09A203.99%20198.09%2076.07%2000340.73%20531.98a201.56%20201.56%200%200057.82-221.87C363.52%20204.52%20246.61%20113.87%20211.03%201.72ZM71.45%20372%20210.83%20532.18%20350.27%20372c-67.04%201.6-36.01-14.95-139.36%20102.24l-81.52-93.27c-14.07-10.13-19.25-9.06-57.94-8.97Z%22%20fill%3D%22%23c9754a%22%2F%3E%3Ccircle%20fill%3D%22%23c9754a%22%20cx%3D%22210.88%22%20cy%3D%22341.11%22%20r%3D%2247.35%22%2F%3E%3C%2Fsvg%3E",
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
      const markW = markH * (427 / 597); // the mark's own viewBox aspect
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
