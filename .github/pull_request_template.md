## What changed

-

## Checks

- [ ] `python -m compileall -q tokenizer train.py benchmark data engine experiments`
- [ ] `python train.py --quick --no-chat`
- [ ] C engine builds with `gcc -std=c11 -O2 engine/engine.c -lm`

## Notes

-
