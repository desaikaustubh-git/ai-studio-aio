import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// After an AI Studio H3 node runs, show its text output(s) in a read-only box on
// the node. Each entry lists the message keys to concatenate, in display order.
const SHOW = {
  AIStudioH3PromptWriter: ["info", "h3_prompt"],
  AIStudioH3MultiShotRender: ["text"],
  AIStudioH3ScriptWriter: ["info", "script"],
  AIStudioH3StoryExpander: ["info", "story"],
  AIStudioH3Director: ["info", "shot_breakdown"],
  AIStudioH3ScriptReview: ["info", "review_notes", "revised_script"],
  AIStudioH3RefImages: ["info", "ref_map"],
  AIStudioH3VideoReview: ["report"],
};

app.registerExtension({
  name: "aistudio.h3.outputs",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const keys = SHOW[nodeData?.name];
    if (!keys) return;

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);

      const join = (v) => (Array.isArray(v) ? v.join("") : v ?? "");
      const text = keys
        .map((k) => join(message?.[k]))
        .filter((s) => s && s.length)
        .join("\n\n----------------------------------------\n\n");
      if (!text) return;

      let box = this.widgets?.find((w) => w.name === "output");
      if (!box) {
        box = ComfyWidgets["STRING"](
          this,
          "output",
          ["STRING", { multiline: true }],
          app
        ).widget;
        box.inputEl.readOnly = true;
        box.inputEl.style.opacity = "0.85";
        box.serializeValue = () => undefined; // don't bloat the saved workflow
      }
      box.value = text;

      this.setDirtyCanvas?.(true, true);
      requestAnimationFrame(() => {
        const sz = this.computeSize();
        this.setSize([Math.max(this.size[0], sz[0]), Math.max(this.size[1], sz[1])]);
        app.graph.setDirtyCanvas(true, true);
      });
    };
  },
});
