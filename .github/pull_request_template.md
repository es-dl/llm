## What changed

-

## Checks

- [ ] `pip install -r requirements-dev.txt`
- [ ] `python -m compileall -q tokenizer train.py inference.py benchmarks data engine src`
- [ ] `python train.py --quick --no-chat`
- [ ] C engine builds with `gcc -std=c11 -O2 engine/engine.c -lm`

## Notes

-
