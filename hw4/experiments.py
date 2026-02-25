import timm
import torch
import torch.nn as nn
from torch.optim import AdamW

from hw3.hw1.train import evaluate_with_metrics

from .trainer import DistillationTrainer, train_epoch


def build_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)


def run_single_experiment(args, experiment_id: int, train_loader, test_loader, device: torch.device) -> dict:
    teacher = build_model(args.teacher, num_classes=10, pretrained=True).to(device)
    student = build_model(args.student, num_classes=10, pretrained=True).to(device)

    if args.teacher_checkpoint:
        teacher.load_state_dict(torch.load(args.teacher_checkpoint, map_location=device), strict=False)
    if args.student_checkpoint:
        student.load_state_dict(torch.load(args.student_checkpoint, map_location=device), strict=False)

    ce = nn.CrossEntropyLoss()
    teacher_loss, teacher_acc, teacher_f1 = evaluate_with_metrics(teacher, test_loader, ce, device)
    student_loss, student_acc, student_f1 = evaluate_with_metrics(student, test_loader, ce, device)

    print(f"\n===== Experiment {experiment_id} =====")
    print("== Baseline metrics before distillation ==")
    print(f"Teacher  | loss={teacher_loss:.4f}, acc={teacher_acc:.2f}, f1={teacher_f1:.4f}")
    print(f"Student  | loss={student_loss:.4f}, acc={student_acc:.2f}, f1={student_f1:.4f}")

    trainer = DistillationTrainer(
        teacher=teacher,
        student=student,
        device=device,
        alpha=args.alpha,
        temperature=args.temperature,
        feature_weight=args.feature_weight,
        experiment=experiment_id,
    )

    params = list(student.parameters())
    if trainer.regressor is not None:
        params += list(trainer.regressor.parameters())
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    best_acc = -1.0
    best_f1 = -1.0
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_stats = train_epoch(trainer, train_loader, optimizer, device)
        eval_loss, eval_acc, eval_f1 = evaluate_with_metrics(student, test_loader, ce, device)

        if eval_acc > best_acc:
            best_acc = eval_acc
        if eval_f1 > best_f1:
            best_f1 = eval_f1
        if eval_loss < best_loss:
            best_loss = eval_loss

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_stats['loss']:.4f} "
            f"(ce={train_stats['ce']:.4f}, kd={train_stats['kd']:.4f}, feature={train_stats['feature']:.4f}) | "
            f"val_loss={eval_loss:.4f}, val_acc={eval_acc:.2f}, val_f1={eval_f1:.4f}"
        )

    final_loss, final_acc, final_f1 = evaluate_with_metrics(student, test_loader, ce, device)
    print("== Final student metrics after distillation ==")
    print(f"Student* | loss={final_loss:.4f}, acc={final_acc:.2f}, f1={final_f1:.4f}")

    return {
        "experiment": experiment_id,
        "teacher_loss": teacher_loss,
        "teacher_acc": teacher_acc,
        "teacher_f1": teacher_f1,
        "student_before_loss": student_loss,
        "student_before_acc": student_acc,
        "student_before_f1": student_f1,
        "student_after_loss": final_loss,
        "student_after_acc": final_acc,
        "student_after_f1": final_f1,
        "best_val_loss": best_loss,
        "best_val_acc": best_acc,
        "best_val_f1": best_f1,
        "acc_gain": final_acc - student_acc,
        "f1_gain": final_f1 - student_f1,
    }


def print_summary_table(rows: list[dict]):
    print("\n===== Distillation Summary =====")
    print("exp | before_acc | after_acc | gain_acc | before_f1 | after_f1 | gain_f1 | best_acc | best_f1")
    for row in rows:
        print(
            f"{row['experiment']:>3} | "
            f"{row['student_before_acc']:>10.2f} | "
            f"{row['student_after_acc']:>9.2f} | "
            f"{row['acc_gain']:>8.2f} | "
            f"{row['student_before_f1']:>9.4f} | "
            f"{row['student_after_f1']:>8.4f} | "
            f"{row['f1_gain']:>7.4f} | "
            f"{row['best_val_acc']:>8.2f} | "
            f"{row['best_val_f1']:>7.4f}"
        )