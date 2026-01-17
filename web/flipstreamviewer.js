import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

app.registerExtension({
    name: "FlipStreamViewer.RunOnce",
    async setup() {
        api.addEventListener("FlipStreamViewer_run_once", async () => {
            await api.interrupt();
            await api.fetchApi("/queue", {
                method: "POST",
                body: JSON.stringify({ clear: true }),
            });
            await app.queuePrompt(0, 1);
        });
    }
});

app.registerExtension({
    name: "FlipStreamViewer.FlipStreamChatJson",
    async beforeRegisterNodeDef(t, d) {
        const config = {
            "FlipStreamGet":      { in: 0,  out: 2, range: 20, iname:'label', oname:'value', useo: true },
            "FlipStreamChatJson": { in: 9, out: 3, range: 20, iname:'label', oname:'value', useo: true },
            "FlipStreamTextConcat": { in: 2, out: 1, range: 20, iname:'text', oname:'', useo: false }
        };
        const CFG = config[d.name];
        if (!CFG) return;

        const sync = (n) => {
            const w = n.widgets;
            if (CFG.useo) {
                while (n.outputs.length > w.length + CFG.out - CFG.in) {
                    if (n.outputs.length < CFG.out || n.outputs[n.outputs.length - 1]?.links?.length) break;
                    n.removeOutput(n.outputs.length - 1);
                    n.outputs_dirty = true;
                }
            }
            if (w.length < CFG.in + CFG.range && (w.at(-1)?.value || n.inputs[w.length - 1]?.link !== null)) {
                if (CFG.useo) {
                    n.addOutput(`${CFG.oname}${w.length - CFG.in}`, "*");
                    n.outputs_dirty = true;
                }
                const inputName = `${CFG.iname}${w.length - CFG.in}`;
                n.addWidget("string", inputName, "", () => {}, { forceInput: true });
                n.inputs_dirty = true;
            }
            while (w.length > CFG.in + 1 && (!w.at(-1).value && n.inputs[w.length - 1]?.link === null) && (!w.at(-2).value && n.inputs[w.length - 2]?.link === null)) {
                if (n.outputs[w.length + CFG.out - CFG.in - 1]?.links?.length) break;
                if (CFG.useo) {
                    n.removeOutput(w.length + CFG.out - CFG.in - 1);
                    n.outputs_dirty = true;
                }
                w.pop();
                n.inputs_dirty = true;
            }
            n.setSize(n.computeSize());
        };

        t.prototype.onNodeCreated = function() { sync(this); };
        t.prototype.onWidgetChanged = function() { sync(this); };
        t.prototype.onConnectionsChange = function(type, slot, connected, info) { sync(this);};

        t.prototype.onConfigure = function(c) {
            c.widgets_values.forEach((v, i) => { if (this.widgets[i]) this.widgets[i].value = v; sync(this);});  
        };
    }
});
