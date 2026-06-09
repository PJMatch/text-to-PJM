#include "PJMPlayerComponent.h"
#include "Components/SkeletalMeshComponent.h"
#include "Animation/AnimInstance.h"
#include "Animation/AnimMontage.h"
#include "TimerManager.h"
#include "Engine/World.h"

UPJMPlayerComponent::UPJMPlayerComponent()
{
    PrimaryComponentTick.bCanEverTick = false;
}

/**
 * Initializes the animation player component.
 *
 * If TargetBodyMesh is not assigned manually, the component tries to find
 * a skeletal mesh component on its owner actor. Then it retrieves the
 * animation instance used for playing montage animations.
 */
void UPJMPlayerComponent::BeginPlay()
{
    Super::BeginPlay();

    if (!TargetBodyMesh)
    {
        AActor* Owner = GetOwner();
        if (Owner)
        {
            TargetBodyMesh = Owner->FindComponentByClass<USkeletalMeshComponent>();
        }
    }

    if (!TargetBodyMesh)
    {
        UE_LOG(LogTemp, Error, TEXT("PJMPlayerComponent: TargetBodyMesh is not assigned."));
        return;
    }

    AnimInstance = TargetBodyMesh->GetAnimInstance();

    if (!AnimInstance)
    {
        UE_LOG(LogTemp, Error, TEXT("PJMPlayerComponent: No AnimInstance on TargetBodyMesh."));
    }
}


/**
 * Starts playing a sequence of animations by their names.
 *
 * The function clears any previous playback, looks up every animation name
 * in the AnimationMap, adds valid animations to the queue, and starts playback
 * from the first available animation.
 *
 * @param AnimationNames List of animation keys to play in order.
 */
void UPJMPlayerComponent::PlayAnimationNames(const TArray<FString>& AnimationNames)
{
    if (!TargetBodyMesh)
    {
        UE_LOG(LogTemp, Error, TEXT("PJMPlayerComponent: TargetBodyMesh is null."));
        return;
    }

    AnimInstance = TargetBodyMesh->GetAnimInstance();
    if (!AnimInstance)
    {
        UE_LOG(LogTemp, Error, TEXT("PJMPlayerComponent: AnimInstance is null."));
        return;
    }

    StopPlayback();
    Queue.Empty();

    for (const FString& Name : AnimationNames)
    {
        const TObjectPtr<UAnimSequence>* Found = AnimationMap.Find(Name);
        if (Found && *Found)
        {
            Queue.Add(*Found);
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("Missing animation for key '%s'"), *Name);
        }
    }

    if (Queue.Num() == 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("Queue is empty."));
        return;
    }

    PlayNext();
}


/**
 * Plays the next animation from the queue.
 *
 * The function removes the first animation from the queue, creates a dynamic
 * montage from it, and schedules a timer that triggers playback of the next
 * animation after the current one is almost finished.
 */
void UPJMPlayerComponent::PlayNext()
{
    if (!AnimInstance)
    {
        return;
    }

    if (Queue.Num() == 0)
    {
        UE_LOG(LogTemp, Log, TEXT("Playback finished."));
        return;
    }

    UAnimSequence* Sequence = Queue[0];
    Queue.RemoveAt(0);

    if (!Sequence)
    {
        OnCurrentFinished();
        return;
    }

    UAnimMontage* Montage = AnimInstance->PlaySlotAnimationAsDynamicMontage(
        Sequence,
        SlotName,
        BlendInTime,
        BlendOutTime,
        PlayRate,
        1
    );

    if (!Montage)
    {
        UE_LOG(LogTemp, Warning, TEXT("Failed to play '%s'"), *Sequence->GetName());
        OnCurrentFinished();
        return;
    }

    const float Duration = Sequence->GetPlayLength() / FMath::Max(PlayRate, 0.01f);
    const float Delay = FMath::Max(Duration - BlendOutTime, 0.01f);

    GetWorld()->GetTimerManager().SetTimer(
        NextTimerHandle,
        this,
        &UPJMPlayerComponent::OnCurrentFinished,
        Delay,
        false
    );
}

/**
 * Handles the end of the currently played animation.
 *
 * This function is called by the timer after the current animation finishes
 * and immediately starts the next animation from the queue.
 */
void UPJMPlayerComponent::OnCurrentFinished()
{
    PlayNext();
}

/**
 * Stops the current animation playback.
 *
 * The function clears the playback timer, empties the animation queue,
 * and stops all currently playing montages on the animation instance.
 */
void UPJMPlayerComponent::StopPlayback()
{
    if (GetWorld())
    {
        GetWorld()->GetTimerManager().ClearTimer(NextTimerHandle);
    }

    Queue.Empty();

    if (AnimInstance)
    {
        AnimInstance->StopAllMontages(0.15f);
    }
}