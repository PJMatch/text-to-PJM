#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Animation/AnimSequence.h"
#include "PJMPlayerComponent.generated.h"

class USkeletalMeshComponent;
class UAnimInstance;

UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class PJMATCH_API UPJMPlayerComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UPJMPlayerComponent();

protected:
    virtual void BeginPlay() override;

public:
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "PJM")
    TObjectPtr<USkeletalMeshComponent> TargetBodyMesh = nullptr;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "PJM")
    TMap<FString, TObjectPtr<UAnimSequence>> AnimationMap;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "PJM")
    FName SlotName = TEXT("PJM_Slot");

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "PJM")
    float BlendInTime = 0.12f;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "PJM")
    float BlendOutTime = 0.12f;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "PJM")
    float PlayRate = 1.0f;

    UFUNCTION(BlueprintCallable, Category = "PJM")
    void PlayAnimationNames(const TArray<FString>& AnimationNames);

    UFUNCTION(BlueprintCallable, Category = "PJM")
    void StopPlayback();

private:
    UPROPERTY()
    TObjectPtr<UAnimInstance> AnimInstance = nullptr;

    UPROPERTY()
    TArray<TObjectPtr<UAnimSequence>> Queue;

    FTimerHandle NextTimerHandle;

    void PlayNext();
    void OnCurrentFinished();
};