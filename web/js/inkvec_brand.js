/**
 * Inkvec branding: the Inkvec mark (the ink drop and its dot) above the INKVEC wordmark,
 * drawn in a reserved strip at the bottom of every Inkvec node. Copper on the node's own
 * ground, so it reads on light and dark ComfyUI themes alike. Purely visual -- remove
 * this file (and WEB_DIRECTORY in __init__.py) and the nodes work exactly as before.
 */
import { app } from "../../scripts/app.js";

const BRAND = {
  mark: "data:image/svg+xml;charset=utf-8,%3Csvg%20width%3D%221350%22%20height%3D%221800%22%20viewBox%3D%220%200%201350%201800%22%20fill%3D%22none%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%3E%3Cpath%20fill-rule%3D%22evenodd%22%20clip-rule%3D%22evenodd%22%20d%3D%22M665.2%2044.6%20582.8%20154.7c-409%20546.9-596.5%20851.2-490%201174.5C196%201643%20542.3%201838.1%20869.5%201714.5A624.2%20624.2%200%20001252%201278.1C1323.2%20973%201138.4%20674.8%20734.9%20137.7Zm355.2%201109.3-100.3.1A255.1%20255.1%200%2000652.3%20898.7%20254.9%20254.9%200%2000410.9%201153l-100.4.9%20354.9%20538.4Z%22%20fill%3D%22%23c9754a%22%2F%3E%3Ccircle%20fill%3D%22%23c9754a%22%20cx%3D%22665.2%22%20cy%3D%221153.6%22%20r%3D%22159.9%22%2F%3E%3C%2Fsvg%3E",
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
      const markW = markH * (1350 / 1800); // the mark's own viewBox aspect
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
