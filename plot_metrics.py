import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
df = pd.read_csv('metrics_all_frames.csv')
df = df[df['method'].isin(['PB', 'deepmag', 'flowmag', 'EulerMormer', 'LS'])]
df.loc[df['method']=='LS', 'method'] = 'PBLS (ours)'
sns.set_theme(style="whitegrid", context="talk") # "talk" = bom equilíbrio
plt.figure(figsize=(18, 14), dpi=120)
sns.scatterplot( data=df, x='dw-ms-ssim', y='flow_amplitude', hue='method', style='method', s=120, alpha=0.8, edgecolor='gray')
# Ajuste das legendas
plt.legend(title='Método', fontsize=11, title_fontsize=12, loc='best')
# Ajuste de labels e título
plt.xlabel('DW-MS-SSIM', fontsize=13)
plt.ylabel('Flow Amplitude', fontsize=13)
plt.title('Comparing methods', fontsize=15)
plt.tight_layout()# Salvar como PNG
plt.savefig('comparacao_metodos.png', dpi=300, bbox_inches='tight')
plt.close()
plt.show()


df_video_mean = ( df.groupby(['method', 'video_name'], as_index=False) .agg({'dw-ms-ssim': 'mean', 'flow_amplitude': 'mean'}))
plt.figure(figsize=(8, 6), dpi=120)
sns.scatterplot( data=df_video_mean, x='dw-ms-ssim', y='flow_amplitude', hue='method', style='method', s=120, alpha=0.8, edgecolor='gray')
if 1: linestyles = ["-", "--", "-.", ":"] # sólida, tracejada, ponto-traço, pontilhada 
for i, method in enumerate(df_video_mean["video_name"].unique()): 
    subdf = df_video_mean[df_video_mean["video_name"] == method] 
    subdf_sorted = subdf.sort_values("dw-ms-ssim") 
    plt.plot( subdf_sorted["dw-ms-ssim"], subdf_sorted["flow_amplitude"], linestyle=linestyles[i % len(linestyles)], color="gray", alpha=0.5, zorder=0, linewidth=1 )
#plt.xscale('log') # apenas eixo X log#plt.yscale('log') # descomente se quiser log também no Y
plt.ylim(0.0, 4.0) # (min, max) — use None para não limitar
plt.xlim(0.90, 1.0) # (min, max) — use None para não limitar
plt.xlabel('DW-MS-SSIM')
plt.ylabel('Flow Amplitude')
plt.title('Comparing methods', fontsize=15)
plt.legend(title='Método', fontsize=11)
plt.tight_layout()
plt.savefig('comparacao_metodos_media.png', dpi=300, bbox_inches='tight')
plt.close()
plt.show()

### Violin plot da métrica1 por método
# Configurar estilo e contexto
sns.set_theme(style="whitegrid", context="talk")
plt.figure(figsize=(10,8))
# Violin plot
#if 0: 
#    sns.violinplot( data=df, x="method", y="dw-ms-ssim", inner="quartile", # mostra quartis dentro do violino palette="pastel" )

sns.boxplot(x="method", y="dw-ms-ssim", data=df, palette="pastel")
#sns.stripplot(x="method", y="dw-ms-ssim", data=df, color="black", size=4, jitter=True)

plt.ylim(0.79, 1.01) # (min, max) — use None para não limitar
plt.xlabel("Method")
plt.ylabel("DW-MS-SSIM")
plt.title("Distribution of DW-MS-SSIM by Method")
plt.tight_layout()
plt.savefig("boxplot.png", dpi=300)
plt.close()
plt.show()