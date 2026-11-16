# Chrysalis on a Mac with Apple Silicon and 32 GB or more

```sh
xcode-select --install
git clone https://github.com/michaelDinelle/chrysalis-inference
cd chrysalis-inference
./chrysalis.sh
```

If the first start says Metal's compiler is missing:

```sh
xcodebuild -downloadComponent MetalToolchain
```

<p align="center"><img src="../media/mac-run.png" width="760" alt="The chat on a Mac answering what a chrysalis is, with Qwen3.8-27B, its time to first token and its speed under the reply"></p>
<p align="center"><img src="../media/mac-picture.png" width="760" alt="A chart dropped into the chat on a Mac, and Qwen3.8-27B saying what it shows"></p>

The release, [Chrysalis.dmg](https://github.com/michaelDinelle/chrysalis-inference/releases/latest/download/Chrysalis.dmg),
is a laptop's chat alone, for Chrysalis on another computer: open it, drag Chrysalis to Applications and open it there.
