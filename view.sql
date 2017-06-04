create OR REPLACE VIEW psl as SELECT (e.page_id*10+e.tab_number) as id, e.discussion_id, e.post_id, e.presented_quote, e.presented_response, e.topic,
  r.attacking_respectful, r.disagree_agree, r.emotion_fact, r.nasty_nice, r.sarcasm_yes,
  p.author_id, q.source_post_id, s.topic_stance_votes_1, s.topic_stance_votes_2
  from mturk_2010_qr_entry e
  join mturk_2010_qr_task1_average_response r
    on e.page_id = r.page_id AND e.tab_number = r.tab_number
  join post p
    on e.discussion_id = p.discussion_id AND e.post_id = p.post_id
  join quote q
    on e.discussion_id = q.discussion_id AND e.post_id = q.post_id AND e.quote_index = q.quote_index
  left join mturk_author_stance s
    on e.discussion_id = s.discussion_id and p.author_id = s.author_id